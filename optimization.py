import torch
import torch.nn.functional as F
import numpy as np
import sys

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

def evaluate_full_field(target_points, base_points, c_grad, p_grad, w_scal, p_scal):
    """
    计算总势函数场 f(x) = Phi_grad(x) + S_res(x)
    """
    N_base = base_points.shape[0]
    
    delta = target_points.unsqueeze(1) - base_points.unsqueeze(0)
    r2 = torch.sum(delta**2, dim=-1)
    r = torch.sqrt(r2)
    log_r = torch.where(r < 1e-12, torch.zeros_like(r), torch.log(r))
    
    # 1. 计算无旋势函数 Phi_grad(x)
    # nabla phi = r^2(4 log r + 1) * delta
    grad_phi_blocks = r2.unsqueeze(-1) * (4 * log_r.unsqueeze(-1) + 1.0) * delta
    c_reshaped = c_grad.view(N_base, 2)
    
    # RBF 部分
    Phi_rbf = torch.einsum('tbd,bd->t', grad_phi_blocks, c_reshaped)
    
    # 多项式部分: p_0*x + p_1*y + p_2*x*y + 0.5*p_3*x^2 + 0.5*p_4*y^2
    x = target_points[:, 0]
    y = target_points[:, 1]
    Phi_poly = p_grad[0]*x + p_grad[1]*y + p_grad[2]*x*y + 0.5*p_grad[3]*x**2 + 0.5*p_grad[4]*y**2
    
    Phi = Phi_rbf + Phi_poly
    
    # 2. 计算残差标量场 S_res(x)
    S_phi = r2 * log_r  # r^2 log r
    S_val = torch.matmul(S_phi, w_scal).squeeze()
    # P_1: a_0 + a_1*x + a_2*y
    x = target_points[:, 0]
    y = target_points[:, 1]
    S_poly = p_scal[0] + p_scal[1]*x + p_scal[2]*y
    
    return Phi + (S_val + S_poly)

def opt(points_np, values_np, init_grads_np, num_iter=500, lr=1e-2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    print(f"Device: {device}, dtype: {dtype}")
    
    points = torch.tensor(points_np, dtype=dtype, device=device)
    values = torch.tensor(values_np, dtype=dtype, device=device).squeeze()
    N = points.shape[0]
    
    # 依然保留你的精妙主意：Eikonal 硬约束，只优化 theta
    init_angles = np.arctan2(init_grads_np[:, 1], init_grads_np[:, 0])
    theta = torch.tensor(init_angles, dtype=dtype, device=device, requires_grad=True)
    
    # 表面加权
    # surface_weights = torch.exp(-5.0 * torch.abs(values))
    surface_weights = torch.ones_like(values)

    
    # 预计算两大矩阵
    print("Building two-step matrices...")
    A_grad, A_scalar = build_two_step_macedo_matrices(points)
    LU_grad, pivots_grad = torch.linalg.lu_factor(A_grad)
    LU_scal, pivots_scal = torch.linalg.lu_factor(A_scalar)
    
    # 提取核矩阵用于能量计算
    A_grad_core = A_grad[:2*N, :2*N]
    A_scal_core = A_scalar[:N, :N]
    
    optimizer = torch.optim.Adam([theta], lr=lr)
    
    print("Starting optimization...")
    for i in range(num_iter):
        optimizer.zero_grad()
        
        grads_x = torch.cos(theta)
        grads_y = torch.sin(theta)
        grads = torch.stack([grads_x, grads_y], dim=1)
        
        # 显式验证梯度长度为 1（Eikonal 约束）
        grad_norms = torch.sqrt(grads_x**2 + grads_y**2)
        loss_eikonal = torch.mean((grad_norms - 1.0)**2)  # 约束 |grad| = 1
        
        # --- 两步法 ---
        # Step 1: 解 Curl-free 梯度系数
        y_grad = torch.cat([grads.view(-1, 1), torch.zeros((5, 1), device=device, dtype=dtype)], dim=0)
        coeffs_grad_all = torch.linalg.lu_solve(LU_grad, pivots_grad, y_grad)
        c_grad = coeffs_grad_all[:2*N]
        p_grad = coeffs_grad_all[2*N:]
        
        # Step 2: 算残差并求解标量系数
        # 注意：这里用 points 评估自身，求出理论 SDF 值
        Phi_at_points = evaluate_full_field(points, points, c_grad, p_grad, 
                                            torch.zeros(N, 1, device=device, dtype=dtype), 
                                            torch.zeros(3, 1, device=device, dtype=dtype))
        residual = values - Phi_at_points
        
        y_scal = torch.cat([residual.unsqueeze(1), torch.zeros((3, 1), device=device, dtype=dtype)], dim=0)
        coeffs_scal_all = torch.linalg.lu_solve(LU_scal, pivots_scal, y_scal)
        w_scal = coeffs_scal_all[:N]
        p_scal = coeffs_scal_all[N:]
        
        # --- 终极目标：单侧投影 Loss ---
        p_proj = points - values.unsqueeze(1) * grads
        projected_values = evaluate_full_field(p_proj, points, c_grad, p_grad, w_scal, p_scal)
        
        # 单侧 ReLU 惩罚
        violation = F.relu(torch.sign(values) * projected_values**2)
        loss_proj = torch.mean(surface_weights * (violation**2))
        
        # 插值能量项
        # energy_grad = torch.matmul(c_grad.T, torch.matmul(A_grad_core, c_grad))
        # energy_scal = torch.matmul(w_scal.T, torch.matmul(A_scal_core, w_scal))
        # loss_energy = energy_grad + energy_scal
        loss_energy = 0.0
        
        # 总损失：投影误差 + Eikonal 约束
        # 总损失：投影误差 + Eikonal 约束 + 插值能量
        loss = loss_proj + 100.0 * loss_eikonal + 0.01 * loss_energy
        
        loss.backward()
        optimizer.step()
        
        if (i+1) % 50 == 0:
            print(f"Step {i+1:3d} | Proj: {loss_proj.item():.6e}  Eikonal: {loss_eikonal.item():.6e}   Total: {loss.item():.6e}")
            
    final_grads_x = torch.cos(theta)
    final_grads_y = torch.sin(theta)
    return torch.stack([final_grads_x, final_grads_y], dim=1).detach().cpu().numpy()