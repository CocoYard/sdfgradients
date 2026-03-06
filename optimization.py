import numpy as np
from interpolation import CurlFree_Interpolator
import torch

def iterative_projection(points, values, init_gradients, num_iter=10,
                         num_coarse=24, refine_steps=4, num_refine=12):
    """
    Iteratively refine SDF gradients by projecting sample points onto the zero
    level set of a curl-free interpolant, then finding the best gradient
    direction via sample_best_gradients (coarse sweep + angular refinement).

    Algorithm (each iteration):
      1. Fit a CurlFree_Interpolator with current gradients (use_projection=True).
      2. For each sample point, search over directions to find the one whose
         projection P - s*g lands closest to the zero level set of the
         interpolant (via sample_best_gradients).
      3. Update gradients := best directions found.
      4. Repeat.

    Parameters
    ----------
    points : (N, 2) array
        Sample point coordinates.
    values : (N,) array
        Signed distance values at each sample point.
    init_gradients : (N, 2) array
        Initial unit gradient estimates.
    num_iter : int
        Number of projection-refit iterations (default 10).
    num_coarse : int
        Number of uniformly spaced directions in the coarse sweep (default 24).
    refine_steps : int
        Number of zoom-in refinement iterations (default 4).
    num_refine : int
        Directions evaluated per refinement step (default 12).

    Returns
    -------
    gradients : (N, 2) array
        Refined unit gradient vectors.
    interpolator : CurlFree_Interpolator
        The final fitted interpolator (ready for predict / marching cubes).
    """
    points = np.asarray(points, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64).ravel()
    gradients = np.array(init_gradients, dtype=np.float64, copy=True)

    # Normalize initial gradients
    norms = np.linalg.norm(gradients, axis=1, keepdims=True)
    gradients /= np.maximum(norms, 1e-12)

    for it in range(num_iter):
        # ----- Step 1: Fit interpolant with current gradients -----
        interpolator = CurlFree_Interpolator(use_projection=True)
        interpolator.fit(points, values, gradients)

        # ----- Step 2: Find best gradient via angular search on the interpolant -----
        new_gradients = interpolator.sample_best_gradients(
            points, values,
            num_coarse=num_coarse,
            refine_steps=refine_steps,
            num_refine=num_refine)

        # ----- Convergence diagnostic -----
        cos_sim = np.sum(gradients * new_gradients, axis=1)
        mean_cos = np.mean(cos_sim)
        max_angle_deg = np.degrees(np.arccos(np.clip(np.min(cos_sim), -1, 1)))

        # Projection error: f(P - s*g) should be ~0
        proj_pts = points - values[:, np.newaxis] * new_gradients
        proj_vals = interpolator.predict(proj_pts)
        proj_rmse = np.sqrt(np.mean(proj_vals**2))

        print(f"Iter {it+1:3d} | mean cos_sim: {mean_cos:.6f}  "
              f"max angle change: {max_angle_deg:.2f}\u00b0  "
              f"proj RMSE: {proj_rmse:.6e}")

        gradients = new_gradients

    # Final fit with converged gradients
    interpolator = CurlFree_Interpolator(use_projection=True)
    interpolator.fit(points, values, gradients)

    return gradients, interpolator


def build_two_step_macedo_matrices_with_projection(points, values, init_gradients, min_proj_distance=1e-8):
    """
    Improved two-step matrix construction using projected points to enhance gradient interpolation.
    Projected point position: P_proj = P - S * d
    
    Same strategy as CurlFree_Interpolator in interpolation.py:
    - Filter out projected points too close to existing points
    - Constrain gradients on all base points (original + valid projected)
    - Build a square system (2*N_cf+5, 2*N_cf+5) to avoid singular matrices
    
    Returns: A_grad, A_scalar, all_base_points, valid_mask
    """
    N, d = points.shape
    device, dtype = points.device, points.dtype
    
    # Compute projected points: P_proj = P - S * d
    projected_points = points - values.unsqueeze(1) * init_gradients
    
    # Filter projected points: keep only those far enough from all existing points
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
    # Step 1: Gradient interpolation matrix A_grad (Curl-free)
    # Based on Hessian of PHS phi(r) = r^4 log r
    # Constrain gradients on all N_cf base points -> square system (2*N_cf+5, 2*N_cf+5)
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
    
    # Polynomial constraint matrix (2*N_cf, 5)
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
    ], dim=0)  # (2*N_cf+5, 2*N_cf+5) -- square matrix
    
    # ==========================================
    # Step 2: Residual interpolation matrix A_scalar (original N points only)
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
    
    # Regularization
    A_grad += torch.eye(A_grad.shape[0], device=device, dtype=dtype) * 1e-10
    A_scalar += torch.eye(A_scalar.shape[0], device=device, dtype=dtype) * 1e-10
    
    return A_grad, A_scalar, all_base_points, valid_mask

def build_two_step_macedo_matrices(points):
    """
    Build two-step matrices strictly following the paper.
    Step 1 (gradient): Hessian of PHS kernel phi(r) = r^4 log(r) for gradient interpolation
    Step 2 (residual): TPS kernel phi(r) = r^2 log(r) with P_1 polynomial space (3 bases)
    """
    N, d = points.shape
    device, dtype = points.device, points.dtype
    
    delta = points.unsqueeze(1) - points.unsqueeze(0)
    r2 = torch.sum(delta**2, dim=-1)
    r = torch.sqrt(r2)
    
    # Safe log(r) computation to prevent NaN when r=0
    log_r = torch.where(r < 1e-12, torch.zeros_like(r), torch.log(r))
    
    # ==========================================
    # Step 1: Gradient interpolation matrix A_grad (Curl-free)
    # Based on Hessian of PHS phi(r) = r^4 log r:
    # H = r^2(4 log r + 1)I + (8 log r + 6)(delta @ delta^T)
    # ==========================================
    I = torch.eye(d, device=device, dtype=dtype).view(1, 1, d, d)
    outer_product = torch.einsum('nid,nie->nide', delta, delta)
    
    H_blocks = r2.view(N, N, 1, 1) * (4 * log_r.view(N, N, 1, 1) + 1.0) * I + \
               (8 * log_r.view(N, N, 1, 1) + 6.0) * outer_product
               
    A_grad_core = H_blocks.permute(0, 2, 1, 3).reshape(N * d, N * d)
    
    # Polynomial constraint matrix for gradients (2N, 5), corresponding to P_2 potential without constant term
    P_grad = torch.zeros((2 * N, 5), device=device, dtype=dtype)
    P_grad[0::2, 0] = 1.0         # constant term for px
    P_grad[1::2, 1] = 1.0         # constant term for py
    P_grad[0::2, 2] = points[:, 1]  # x-component of x*y
    P_grad[1::2, 2] = points[:, 0]  # y-component of x*y
    P_grad[0::2, 3] = points[:, 0]  # x^2 coefficient
    P_grad[1::2, 4] = points[:, 1]  # y^2 coefficient
    
    A_grad = torch.cat([
        torch.cat([A_grad_core, P_grad], dim=1),
        torch.cat([P_grad.T, torch.zeros((5, 5), device=device, dtype=dtype)], dim=1)
    ], dim=0)
    
    # ==========================================
    # Step 2: Residual interpolation matrix A_scalar
    # 2D TPS: phi = r^2 log r, P_1 polynomial space (3 bases)
    # ==========================================
    A_scal_core = r2 * log_r  # r^2 log r
    
    # Polynomial tail P_1(x, y) = a_0 + a_1*x + a_2*y
    E_scal = torch.cat([
        torch.ones((N, 1), device=device, dtype=dtype),
        points
    ], dim=1)
    A_scalar = torch.cat([
        torch.cat([A_scal_core, E_scal], dim=1),
        torch.cat([E_scal.T, torch.zeros((3, 3), device=device, dtype=dtype)], dim=1)
    ], dim=0)
    
    # Tiny perturbation to ensure safe LU factorization
    A_grad += torch.eye(A_grad.shape[0], device=device, dtype=dtype) * 1e-10
    A_scalar += torch.eye(A_scalar.shape[0], device=device, dtype=dtype) * 1e-10
    
    return A_grad, A_scalar

def _safe_r_logr(r2):
    """
    Safely compute r and log(r) from r^2, producing no inf/NaN in forward or backward pass.
    For r=0 positions, returns r=0, log_r=0 (kernel values are exactly 0 since
    grad_phi is proportional to delta=0, TPS is proportional to r^2*log_r -> 0*(-inf)=0).
    """
    eps = 1e-24
    r2_safe = r2 + eps                     # Ensure >0, finite sqrt gradient
    r = torch.sqrt(r2_safe)
    log_r = 0.5 * torch.log(r2_safe)       # log(sqrt(r2+eps))
    # Zero out log_r where r is truly ~0 (kernel is 0 on the diagonal anyway)
    mask = (r2 < 1e-20)
    log_r = log_r.masked_fill(mask, 0.0)
    r = r.masked_fill(mask, 0.0)
    return r, log_r

def evaluate_full_field(target_points, base_points, c_grad, p_grad, w_scal, p_scal):
    """
    Evaluate the total potential field f(x) = Phi_grad(x) + S_res(x)
    """
    N_base = base_points.shape[0]
    
    delta = target_points.unsqueeze(1) - base_points.unsqueeze(0)
    r2 = torch.sum(delta**2, dim=-1)
    r, log_r = _safe_r_logr(r2)
    
    # 1. Compute curl-free potential Phi_grad(x)
    grad_phi_blocks = r2.unsqueeze(-1) * (4 * log_r.unsqueeze(-1) + 1.0) * delta
    c_reshaped = c_grad.view(N_base, 2)
    Phi_rbf = torch.einsum('tbd,bd->t', grad_phi_blocks, c_reshaped)
    
    x = target_points[:, 0]
    y = target_points[:, 1]
    Phi_poly = p_grad[0]*x + p_grad[1]*y + p_grad[2]*x*y + 0.5*p_grad[3]*x**2 + 0.5*p_grad[4]*y**2
    Phi = Phi_rbf + Phi_poly
    
    # 2. Compute residual scalar field S_res(x)
    S_phi = r2 * log_r
    S_val = torch.matmul(S_phi, w_scal).squeeze()
    S_poly = p_scal[0] + p_scal[1]*x + p_scal[2]*y
    
    return Phi + (S_val + S_poly)

def evaluate_full_field_with_projection(target_points, base_points, c_grad, p_grad, w_scal, p_scal):
    """
    Evaluate the total potential field f(x) = Phi_grad(x) + S_res(x)
    
    base_points: gradient interpolation base points (original + valid projected)
    w_scal: residual interpolation coefficients, corresponding to the original N points
    """
    N_base = base_points.shape[0]
    N_orig = w_scal.shape[0]
    base_points_orig = base_points[:N_orig]
    
    # --- Gradient potential (using all base points) ---
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
    
    # --- Residual scalar field (original base points only) ---
    delta_orig = target_points.unsqueeze(1) - base_points_orig.unsqueeze(0)
    r2_orig = torch.sum(delta_orig**2, dim=-1) # 2x2
    _, log_r_orig = _safe_r_logr(r2_orig)
    
    S_phi = r2_orig * log_r_orig
    S_val = torch.matmul(S_phi, w_scal).squeeze()
    S_poly = p_scal[0] + p_scal[1]*x + p_scal[2]*y
    
    return Phi + (S_val + S_poly)

def opt(points_np, values_np, init_grads_np, num_iter=500, lr=1e-2, rebuild_every=1, hard_eikonal=False,
        w_proj=1.0, w_smooth=1, w_init=0.01, w_eikonal=0.1, k_neighbors=6):
    """
    Optimize projected point positions to derive SDF gradients.
    
    Instead of optimizing gradient vectors directly, optimize the projected point
    positions P_proj (where P_proj = P - s*g). Gradients are then derived as:
      g = sign(s) * (P - P_proj) / |P - P_proj|
    
    This parameterization tends to produce smoother surfaces because neighboring
    projected points on the zero level set are directly regularized.
    
    Loss function design:
      1. Projection loss: f(P_proj)^2 -> 0
         Projected points should lie on the zero level set.
      2. Gradient smoothness loss (k-NN): sum w_ij (1 - g_i . g_j)^2
         Penalize inconsistent gradient directions derived from projected points.
      3. Initial position anchor: ||P_proj - P_proj_init||^2
         Prevent projected points from drifting far from initial estimate.
      4. Distance constraint: (|P - P_proj| - |s|)^2
         Projected distance should match the SDF value (eikonal consistency).
    
    Parameters:
        w_proj:      Projection loss weight (default 1.0)
        w_smooth:    Smoothness loss weight (default 1)
        w_init:      Initial position anchor weight (default 0.01)
        w_eikonal:   Distance constraint weight (default 0.1)
        k_neighbors: Number of k-NN neighbors for smoothness loss (default 6)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    print(f"Device: {device}, dtype: {dtype}")
    
    points = torch.tensor(points_np, dtype=dtype, device=device)
    values = torch.tensor(values_np, dtype=dtype, device=device).squeeze()
    N = points.shape[0]
    
    # Normalized initial gradients -> compute initial projected points
    init_grads_tensor = torch.tensor(init_grads_np, dtype=dtype, device=device)
    init_grads_tensor = init_grads_tensor / (torch.norm(init_grads_tensor, dim=1, keepdim=True) + 1e-12)
    init_proj_points = points - values.unsqueeze(1) * init_grads_tensor
    
    # sign(s) for gradient direction recovery; for s=0 use +1 as placeholder
    s_sign = torch.sign(values)
    s_sign[s_sign == 0] = 1.0
    s_sign = s_sign.unsqueeze(1)  # (N, 1)
    
    k = min(k_neighbors, N - 1)
    
    def rebuild_knn(p_proj_detached):
        """Rebuild k-NN graph based on projected point positions (on the zero level set)."""
        dists_matrix = torch.cdist(p_proj_detached, p_proj_detached)  # (N, N)
        _, idx = torch.topk(dists_matrix, k + 1, largest=False)
        idx = idx[:, 1:]  # Exclude self -> (N, k)
        # Inverse distance weights: closer neighbors have more influence
        dists = torch.gather(dists_matrix, 1, idx)  # (N, k)
        weights = 1.0 / (dists + 1e-10)
        weights = weights / weights.sum(dim=1, keepdim=True)  # Normalize
        return idx, weights
    
    # ==========================================
    # Optimization variable: projected point positions
    # ==========================================
    proj_points = init_proj_points.detach().clone().requires_grad_(True)
    opt_params = [proj_points]
    
    # Store initial projected positions for anchor loss
    init_proj_detached = init_proj_points.detach().clone()
    
    # Gentle decay weights: points closer to surface get more weight
    surface_weights = 1.0 / (1.0 + torch.abs(values))
    
    def derive_gradients(p_proj):
        """Derive unit gradient vectors from projected point positions.
        g = sign(s) * (P - P_proj) / |P - P_proj|"""
        direction = points - p_proj  # (N, 2), equals s * g
        dist = torch.norm(direction, dim=1, keepdim=True)  # |s| ideally
        grads_normed = s_sign * direction / (dist + 1e-12)
        return grads_normed, dist.squeeze()
    
    def rebuild_matrices(current_grads_detached):
        """Recompute projected points and factorize matrices using current (detached) gradients."""
        A_g, A_s, base_pts, v_mask = build_two_step_macedo_matrices_with_projection(
            points, values, current_grads_detached)
        LU_g, piv_g = torch.linalg.lu_factor(A_g)
        LU_s, piv_s = torch.linalg.lu_factor(A_s)
        return A_g, A_s, base_pts, v_mask, LU_g, piv_g, LU_s, piv_s
    
    # Initial construction
    print("Building two-step matrices with projection (initial)...")
    with torch.no_grad():
        cur_grads, _ = derive_gradients(proj_points)
        knn_idx, knn_weights = rebuild_knn(proj_points.detach())
    A_grad, A_scalar, all_base_points, valid_mask, LU_grad, pivots_grad, LU_scal, pivots_scal = \
        rebuild_matrices(cur_grads.detach())
    N_cf = all_base_points.shape[0]
    
    optimizer = torch.optim.Adam(opt_params, lr=lr)
    
    print(f"Starting optimization (rebuild every {rebuild_every} steps, k={k})...")
    print(f"Weights: proj={w_proj}, smooth={w_smooth}, init={w_init}, dist={w_eikonal}")
    
    for i in range(num_iter):
        # Recompute projected points and matrix factorization every rebuild_every steps
        if i > 0 and i % rebuild_every == 0:
            with torch.no_grad():
                cur_grads, _ = derive_gradients(proj_points)
                knn_idx, knn_weights = rebuild_knn(proj_points.detach())
            A_grad, A_scalar, all_base_points, valid_mask, LU_grad, pivots_grad, LU_scal, pivots_scal = \
                rebuild_matrices(cur_grads.detach())
            N_cf = all_base_points.shape[0]
        
        optimizer.zero_grad()
        
        # Derive gradients from current projected point positions
        grads_normalized, proj_dists = derive_gradients(proj_points)
        
        # =========================================
        # Loss 1: Distance constraint (|P - P_proj| - |s|)^2
        # Ensures the projected distance matches the SDF value (eikonal consistency)
        # =========================================
        loss_dist = torch.mean((proj_dists - torch.abs(values))**2)
        
        # Constrain gradients on all base points (projected point gradients = corresponding original gradients)
        all_grads = torch.cat([grads_normalized, grads_normalized[valid_mask]], dim=0)  # (N_cf, 2)
        
        # --- Two-step solve for interpolation coefficients ---
        # Step 1: Solve curl-free gradient coefficients
        y_grad = torch.cat([all_grads.view(-1, 1), torch.zeros((5, 1), device=device, dtype=dtype)], dim=0)
        coeffs_grad_all = torch.linalg.lu_solve(LU_grad, pivots_grad, y_grad)
        c_grad = coeffs_grad_all[:2*N_cf]
        p_grad = coeffs_grad_all[2*N_cf:]
        
        # Step 2: Compute residual and solve scalar coefficients
        Phi_at_points = evaluate_full_field_with_projection(
            points, all_base_points, c_grad, p_grad,
            torch.zeros(N, 1, device=device, dtype=dtype),
            torch.zeros(3, 1, device=device, dtype=dtype))
        residual = values - Phi_at_points
        
        y_scal = torch.cat([residual.unsqueeze(1), torch.zeros((3, 1), device=device, dtype=dtype)], dim=0)
        coeffs_scal_all = torch.linalg.lu_solve(LU_scal, pivots_scal, y_scal)
        w_scal = coeffs_scal_all[:N]
        p_scal = coeffs_scal_all[N:]
        
        # =========================================
        # Loss 2: Projection loss f(P_proj)^2 -> 0
        # Projected points should lie on the zero level set
        # =========================================
        projected_values = evaluate_full_field_with_projection(
            proj_points, all_base_points, c_grad, p_grad, w_scal, p_scal)
        loss_proj = torch.mean(surface_weights * projected_values**2)
        
        # =========================================
        # Loss 3: Laplacian smoothness on projected points
        # Each projected point should be close to the centroid of its neighbors
        # on the zero level set -> directly reduces jaggedness/sawteeth
        # =========================================
        neighbor_proj = proj_points[knn_idx]  # (N, k, 2)
        centroid = torch.sum(knn_weights.unsqueeze(-1) * neighbor_proj, dim=1)  # (N, 2)
        laplacian = proj_points - centroid  # deviation from local centroid
        loss_smooth = torch.mean(torch.sum(laplacian**2, dim=1))
        
        # =========================================
        # Loss 4: Initial position anchor (mild regularization)
        # Prevent projected points from drifting far from initial estimate
        # =========================================
        loss_init = torch.mean(torch.sum(
            (proj_points - init_proj_detached)**2, dim=1))
        
        # =========================================
        # Total loss
        # =========================================
        loss = (w_proj * loss_proj + w_smooth * loss_smooth +
                w_init * loss_init + w_eikonal * loss_dist)
        
        loss.backward()
        
        # NaN detection & gradient clipping
        if torch.isnan(proj_points.grad).any():
            print(f"  !! NaN gradient detected at step {i+1}, skipping update")
            proj_points.grad.zero_()
            continue
        torch.nn.utils.clip_grad_norm_([proj_points], max_norm=1.0)
        optimizer.step()
        
        if (i+1) % 100 == 0 or i == 0:
            print(f"Step {i+1:3d} | Proj: {loss_proj.item():.6e}  "
                  f"Smooth: {loss_smooth.item():.6e}  "
                  f"Init: {loss_init.item():.6e}  "
                  f"Dist: {loss_dist.item():.6e}  "
                  f"Total: {loss.item():.6e}")
    
    # Convert optimized projected points to unit gradients
    final_grads, _ = derive_gradients(proj_points)
    return final_grads.detach().cpu().numpy()