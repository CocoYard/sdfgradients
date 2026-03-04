import torch
import torch.nn.functional as F
import numpy as np
import sys

def build_two_step_macedo_matrices_with_projection(points, values, init_gradients, min_proj_distance=1e-8):
    """
    改进版两步法矩阵构建，使用投影点增强梯度插值。
    投影点位置：P_proj = P - S * d
    
    与 interpolation.py 中的 CurlFree_Interpolator 相同策略：
    - 过滤掉与已有点距离太近的投影点
    - 在所有基点（原始 + 有效投影点）上同时约束梯度
    - 构建方阵系统 (2*N_cf+5, 2*N_cf+5)，避免奇异矩阵
    
    返回: A_grad, A_scalar, all_base_points, valid_mask
    """
    N, d = points.shape
    device, dtype = points.device, points.dtype
    
    # 计算投影点：P_proj = P - S * d
    projected_points = points - values.unsqueeze(1) * init_gradients
    
    # 过滤投影点：保留与所有已有点距离足够远的投影点
    valid_mask = torch.ones(N, dtype=torch.bool, device=device)
    for i in range(N):
        dists_to_orig = torch.norm(projected_points[i] - points, dim=1)
        if torch.min(dists_to_orig).item() < min_proj_distance:
            valid_mask[i] = False
            continue
        if torch.any(valid_mask[:i]):
            accepted_proj = projected_points[:i][valid_mask[:i]]
            dists_to_accepted = torch.norm(projected_points[i] - accepted_proj, dim=1)
            if dists_to_accepted.numel() > 0 and torch.min(dists_to_accepted).item() < min_proj_distance:
                valid_mask[i] = False
    
    valid_proj = projected_points[valid_mask]
    all_base_points = torch.cat([points, valid_proj], dim=0)
    N_cf = all_base_points.shape[0]
    # print(f"  Projection: {valid_mask.sum().item()}/{N} projected points accepted, "
    #       f"total CF base points: {N_cf}")
    
    # ==========================================
    # Step 1: 梯度插值矩阵 A_grad (Curl-free)
    # 基于 PHS phi(r) = r^4 log r 的 Hessian
    # 在所有 N_cf 个基点上约束梯度 → 方阵 (2*N_cf+5, 2*N_cf+5)
    # ==========================================
    delta = all_base_points.unsqueeze(1) - all_base_points.unsqueeze(0)
    r2 = torch.sum(delta**2, dim=-1)
    r = torch.sqrt(r2)
    log_r = torch.where(r < 1e-12, torch.zeros_like(r), torch.log(r))
    
    I_mat = torch.eye(d, device=device, dtype=dtype).view(1, 1, d, d)
    outer_product = torch.einsum('nid,nie->nide', delta, delta)
    
    H_blocks = r2.view(N_cf, N_cf, 1, 1) * (4 * log_r.view(N_cf, N_cf, 1, 1) + 1.0) * I_mat + \
               (8 * log_r.view(N_cf, N_cf, 1, 1) + 6.0) * outer_product
    
    A_grad_core = H_blocks.permute(0, 2, 1, 3).reshape(N_cf * d, N_cf * d)
    
    # 多项式约束矩阵 (2*N_cf, 5)
    P_grad = torch.zeros((2 * N_cf, 5), device=device, dtype=dtype)
    P_grad[0::2, 0] = 1.0
    P_grad[1::2, 1] = 1.0
    P_grad[0::2, 2] = all_base_points[:, 1]
    P_grad[1::2, 2] = all_base_points[:, 0]
    P_grad[0::2, 3] = all_base_points[:, 0]
    P_grad[1::2, 4] = all_base_points[:, 1]
    
    A_grad = torch.cat([
        torch.cat([A_grad_core, P_grad], dim=1),
        torch.cat([P_grad.T, torch.zeros((5, 5), device=device, dtype=dtype)], dim=1)
    ], dim=0)  # (2*N_cf+5, 2*N_cf+5) — 方阵
    
    # ==========================================
    # Step 2: 残差插值矩阵 A_scalar（只用原始 N 个点）
    # ==========================================
    delta_orig = points.unsqueeze(1) - points.unsqueeze(0)
    r2_orig = torch.sum(delta_orig**2, dim=-1)
    r_orig = torch.sqrt(r2_orig)
    log_r_orig = torch.where(r_orig < 1e-12, torch.zeros_like(r_orig), torch.log(r_orig))
    
    A_scal_core = r2_orig * log_r_orig
    
    E_scal = torch.cat([
        torch.ones((N, 1), device=device, dtype=dtype),
        points
    ], dim=1)
    A_scalar = torch.cat([
        torch.cat([A_scal_core, E_scal], dim=1),
        torch.cat([E_scal.T, torch.zeros((3, 3), device=device, dtype=dtype)], dim=1)
    ], dim=0)
    
    # 正则化
    A_grad += torch.eye(A_grad.shape[0], device=device, dtype=dtype) * 1e-10
    A_scalar += torch.eye(A_scalar.shape[0], device=device, dtype=dtype) * 1e-10
    
    return A_grad, A_scalar, all_base_points, valid_mask

def build_two_step_macedo_matrices(points):
    """
    严格按照论文构建两步法矩阵。
    第一步（梯度）：PHS 核 phi(r) = r^4 log(r) 的 Hessian 用于梯度插值
    第二步（残差）：TPS 核 phi(r) = r^2 log(r)，对应 P_1 多项式空间（3个基）
    """
    N, d = points.shape
    device, dtype = points.device, points.dtype
    
    delta = points.unsqueeze(1) - points.unsqueeze(0)
    r2 = torch.sum(delta**2, dim=-1)
    r = torch.sqrt(r2)
    
    # 安全的 log(r) 计算，防止 r=0 时出现 NaN
    log_r = torch.where(r < 1e-12, torch.zeros_like(r), torch.log(r))
    
    # ==========================================
    # Step 1: 梯度插值矩阵 A_grad (Curl-free)
    # 基于 PHS phi(r) = r^4 log r 的 Hessian:
    # H = r^2(4 log r + 1)I + (8 log r + 6)(delta @ delta^T)
    # ==========================================
    I = torch.eye(d, device=device, dtype=dtype).view(1, 1, d, d)
    outer_product = torch.einsum('nid,nie->nide', delta, delta)
    
    H_blocks = r2.view(N, N, 1, 1) * (4 * log_r.view(N, N, 1, 1) + 1.0) * I + \
               (8 * log_r.view(N, N, 1, 1) + 6.0) * outer_product
               
    A_grad_core = H_blocks.permute(0, 2, 1, 3).reshape(N * d, N * d)
    
    # 梯度的多项式约束矩阵 (2N, 5)，对应势函数 P_2 去掉常数项
    P_grad = torch.zeros((2 * N, 5), device=device, dtype=dtype)
    P_grad[0::2, 0] = 1.0         # px 的常数项
    P_grad[1::2, 1] = 1.0         # py 的常数项
    P_grad[0::2, 2] = points[:, 1]  # x*y 的 x 分量
    P_grad[1::2, 2] = points[:, 0]  # x*y 的 y 分量
    P_grad[0::2, 3] = points[:, 0]  # x^2 系数
    P_grad[1::2, 4] = points[:, 1]  # y^2 系数
    
    A_grad = torch.cat([
        torch.cat([A_grad_core, P_grad], dim=1),
        torch.cat([P_grad.T, torch.zeros((5, 5), device=device, dtype=dtype)], dim=1)
    ], dim=0)
    
    # ==========================================
    # Step 2: 残差插值矩阵 A_scalar
    # 2D TPS: phi = r^2 log r，P_1 多项式空间（3个基）
    # ==========================================
    A_scal_core = r2 * log_r  # r^2 log r
    
    # 势函数的多项式尾巴 P_1(x, y) = a_0 + a_1*x + a_2*y
    E_scal = torch.cat([
        torch.ones((N, 1), device=device, dtype=dtype),
        points
    ], dim=1)
    A_scalar = torch.cat([
        torch.cat([A_scal_core, E_scal], dim=1),
        torch.cat([E_scal.T, torch.zeros((3, 3), device=device, dtype=dtype)], dim=1)
    ], dim=0)
    
    # 极小扰动保证 LU 分解安全
    A_grad += torch.eye(A_grad.shape[0], device=device, dtype=dtype) * 1e-10
    A_scalar += torch.eye(A_scalar.shape[0], device=device, dtype=dtype) * 1e-10
    
    return A_grad, A_scalar

def _safe_r_logr(r2):
    """
    从 r^2 安全计算 r 和 log(r)，前向和反向传播都不产生 inf/NaN。
    对 r=0 的位置，返回 r=0, log_r=0（核函数值恰好为 0，因为
    grad_phi ∝ delta=0, TPS ∝ r^2*log_r → 0·(-∞)=0）。
    """
    eps = 1e-24
    r2_safe = r2 + eps                     # 保证 >0，sqrt 梯度有限
    r = torch.sqrt(r2_safe)
    log_r = 0.5 * torch.log(r2_safe)       # log(sqrt(r2+eps))
    # 将真正 r≈0 的位置的 log_r 置零（核函数在对角线上本身为 0）
    mask = (r2 < 1e-20)
    log_r = log_r.masked_fill(mask, 0.0)
    r = r.masked_fill(mask, 0.0)
    return r, log_r

def evaluate_full_field(target_points, base_points, c_grad, p_grad, w_scal, p_scal):
    """
    计算总势函数场 f(x) = Phi_grad(x) + S_res(x)
    """
    N_base = base_points.shape[0]
    
    delta = target_points.unsqueeze(1) - base_points.unsqueeze(0)
    r2 = torch.sum(delta**2, dim=-1)
    r, log_r = _safe_r_logr(r2)
    
    # 1. 计算无旋势函数 Phi_grad(x)
    grad_phi_blocks = r2.unsqueeze(-1) * (4 * log_r.unsqueeze(-1) + 1.0) * delta
    c_reshaped = c_grad.view(N_base, 2)
    Phi_rbf = torch.einsum('tbd,bd->t', grad_phi_blocks, c_reshaped)
    
    x = target_points[:, 0]
    y = target_points[:, 1]
    Phi_poly = p_grad[0]*x + p_grad[1]*y + p_grad[2]*x*y + 0.5*p_grad[3]*x**2 + 0.5*p_grad[4]*y**2
    Phi = Phi_rbf + Phi_poly
    
    # 2. 计算残差标量场 S_res(x)
    S_phi = r2 * log_r
    S_val = torch.matmul(S_phi, w_scal).squeeze()
    S_poly = p_scal[0] + p_scal[1]*x + p_scal[2]*y
    
    return Phi + (S_val + S_poly)

def evaluate_full_field_with_projection(target_points, base_points, c_grad, p_grad, w_scal, p_scal):
    """
    计算总势函数场 f(x) = Phi_grad(x) + S_res(x)
    
    base_points: 梯度插值基点（原点 + 有效投影点）
    w_scal: 残差插值系数，对应原始 N 个点
    """
    N_base = base_points.shape[0]
    N_orig = w_scal.shape[0]
    base_points_orig = base_points[:N_orig]
    
    # --- 梯度势函数（使用所有基点）---
    delta = target_points.unsqueeze(1) - base_points.unsqueeze(0)
    r2 = torch.sum(delta**2, dim=-1)
    r, log_r = _safe_r_logr(r2)
    
    grad_phi_blocks = r2.unsqueeze(-1) * (4 * log_r.unsqueeze(-1) + 1.0) * delta
    c_reshaped = c_grad.view(N_base, 2)
    Phi_rbf = torch.einsum('tbd,bd->t', grad_phi_blocks, c_reshaped)
    
    x = target_points[:, 0]
    y = target_points[:, 1]
    Phi_poly = p_grad[0]*x + p_grad[1]*y + p_grad[2]*x*y + 0.5*p_grad[3]*x**2 + 0.5*p_grad[4]*y**2
    Phi = Phi_rbf + Phi_poly
    
    # --- 残差标量场（只用原始基点）---
    delta_orig = target_points.unsqueeze(1) - base_points_orig.unsqueeze(0)
    r2_orig = torch.sum(delta_orig**2, dim=-1) # 2x2
    _, log_r_orig = _safe_r_logr(r2_orig)
    
    S_phi = r2_orig * log_r_orig
    S_val = torch.matmul(S_phi, w_scal).squeeze()
    S_poly = p_scal[0] + p_scal[1]*x + p_scal[2]*y
    
    return Phi + (S_val + S_poly)

def opt(points_np, values_np, init_grads_np, num_iter=500, lr=1e-2, rebuild_every=50, hard_eikonal=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    print(f"Device: {device}, dtype: {dtype}")
    
    points = torch.tensor(points_np, dtype=dtype, device=device)
    values = torch.tensor(values_np, dtype=dtype, device=device).squeeze()
    N = points.shape[0]
    
    if hard_eikonal:
        # Eikonal 硬约束：只优化 theta，梯度自动满足 |g|=1
        init_angles = np.arctan2(init_grads_np[:, 1], init_grads_np[:, 0])
        theta = torch.tensor(init_angles, dtype=dtype, device=device, requires_grad=True)
        opt_params = [theta]
    else:
        # 软约束：直接优化梯度向量，用 Eikonal loss 惩罚偏离单位长度
        grads_param = torch.tensor(init_grads_np, dtype=dtype, device=device, requires_grad=True)
        opt_params = [grads_param]
    
    surface_weights = torch.ones_like(values)
    
    def rebuild_matrices(current_grads_detached):
        """用当前（detached）梯度重新计算投影点并分解矩阵。"""
        A_g, A_s, base_pts, v_mask = build_two_step_macedo_matrices_with_projection(
            points, values, current_grads_detached)
        LU_g, piv_g = torch.linalg.lu_factor(A_g)
        LU_s, piv_s = torch.linalg.lu_factor(A_s)
        return A_g, A_s, base_pts, v_mask, LU_g, piv_g, LU_s, piv_s
    
    # 初始构建
    print("Building two-step matrices with projection (initial)...")
    with torch.no_grad():
        if hard_eikonal:
            cur_grads = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
        else:
            cur_grads = grads_param.detach().clone()
    A_grad, A_scalar, all_base_points, valid_mask, LU_grad, pivots_grad, LU_scal, pivots_scal = \
        rebuild_matrices(cur_grads)
    N_cf = all_base_points.shape[0]
    A_grad_core = A_grad[:2*N_cf, :2*N_cf]
    A_scal_core = A_scalar[:N, :N]
    
    optimizer = torch.optim.Adam(opt_params, lr=lr)
    
    print(f"Starting optimization (rebuild matrices every {rebuild_every} steps, "
          f"hard_eikonal={hard_eikonal})...")
    for i in range(num_iter):
        # 每隔 rebuild_every 步用当前梯度重新计算投影点和矩阵分解
        if i > 0 and i % rebuild_every == 0:
            with torch.no_grad():
                if hard_eikonal:
                    cur_grads = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
                else:
                    cur_grads = grads_param.detach().clone()
            A_grad, A_scalar, all_base_points, valid_mask, LU_grad, pivots_grad, LU_scal, pivots_scal = \
                rebuild_matrices(cur_grads)
            N_cf = all_base_points.shape[0]
            A_grad_core = A_grad[:2*N_cf, :2*N_cf]
            A_scal_core = A_scalar[:N, :N]
        
        optimizer.zero_grad()
        
        if hard_eikonal:
            grads_x = torch.cos(theta)
            grads_y = torch.sin(theta)
            grads = torch.stack([grads_x, grads_y], dim=1)
        else:
            grads = grads_param
            grads_x = grads[:, 0]
            grads_y = grads[:, 1]
        
        # Eikonal 约束
        grad_norms = torch.sqrt(grads_x**2 + grads_y**2)
        loss_eikonal = torch.mean((grad_norms - 1.0)**2)
        
        # 在所有基点上约束梯度（投影点梯度 = 对应原始点梯度）
        all_grads = torch.cat([grads, grads[valid_mask]], dim=0)  # (N_cf, 2)
        
        # --- 两步法 ---
        # Step 1: 解 Curl-free 梯度系数
        y_grad = torch.cat([all_grads.view(-1, 1), torch.zeros((5, 1), device=device, dtype=dtype)], dim=0)
        coeffs_grad_all = torch.linalg.lu_solve(LU_grad, pivots_grad, y_grad)
        c_grad = coeffs_grad_all[:2*N_cf]
        p_grad = coeffs_grad_all[2*N_cf:]
        
        # Step 2: 算残差并求解标量系数
        Phi_at_points = evaluate_full_field_with_projection(
            points, all_base_points, c_grad, p_grad,
            torch.zeros(N, 1, device=device, dtype=dtype),
            torch.zeros(3, 1, device=device, dtype=dtype))
        residual = values - Phi_at_points
        
        y_scal = torch.cat([residual.unsqueeze(1), torch.zeros((3, 1), device=device, dtype=dtype)], dim=0)
        coeffs_scal_all = torch.linalg.lu_solve(LU_scal, pivots_scal, y_scal)
        w_scal = coeffs_scal_all[:N]
        p_scal = coeffs_scal_all[N:]
        
        # --- 单侧投影 Loss ---
        p_proj = points - values.unsqueeze(1) * grads
        projected_values = evaluate_full_field_with_projection(
            p_proj, all_base_points, c_grad, p_grad, w_scal, p_scal)
        
        # 单侧惩罚：sign(s) * f(proj) > 0 → 违反（投影点应该穿过零等值面）
        signed_violation = torch.sign(values) * projected_values
        violation = F.relu(signed_violation)
        loss_proj = torch.mean(surface_weights * (violation**2))
        
        # 插值能量项（clamp 到非负，PHS 核矩阵是条件正定，提取子块可能为负）
        # energy_grad = torch.matmul(c_grad.T, torch.matmul(A_grad_core, c_grad)).squeeze()
        # energy_scal = torch.matmul(w_scal.T, torch.matmul(A_scal_core, w_scal)).squeeze()
        # loss_energy = torch.clamp(energy_grad, min=0.0) + torch.clamp(energy_scal, min=0.0)
        loss_energy = 0.0
        
        # 总损失：投影误差 + Eikonal 约束 + 插值能量
        if hard_eikonal:
            loss = loss_proj + 0.01 * loss_energy
        else:
            loss = loss_proj + 0.01 * loss_energy + 0.1 * loss_eikonal
        
        loss.backward()
        
        # NaN 检测 & 梯度裁剪
        param = theta if hard_eikonal else grads_param
        if torch.isnan(param.grad).any():
            print(f"  !! NaN gradient detected at step {i+1}, skipping update")
            param.grad.zero_()
            continue
        torch.nn.utils.clip_grad_norm_([param], max_norm=1.0)
        optimizer.step()
        
        if (i+1) % 500 == 0 or i == 0:
            print(f"Step {i+1:3d} | Proj: {loss_proj.item():.6e}  Eikonal: {loss_eikonal.item():.6e}  Energy: {loss_energy:.6e}  Total: {loss.item():.6e}")
            
    if hard_eikonal:
        final_grads_x = torch.cos(theta)
        final_grads_y = torch.sin(theta)
        return torch.stack([final_grads_x, final_grads_y], dim=1).detach().cpu().numpy()
    else:
        return grads_param.detach().cpu().numpy()