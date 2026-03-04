import numpy as np
from scipy.spatial.distance import cdist

class CurlFree_Interpolator:
    """
    基于旋度自由 RBF (Curl-Free RBF) 和标量薄板样条 (Thin-Plate Spline) 
    的 SDF 全局插值器。
    
    当 use_projection=True 时，会将原始采样点沿梯度方向投影到表面附近，
    作为额外的梯度约束点，增强插值精度。
    """
    def __init__(self, use_projection=False, min_proj_distance=1e-8):
        """
        Parameters:
            use_projection: 是否使用投影点增强梯度插值
            min_proj_distance: 投影点与所有已有点的最小距离阈值，
                               低于此距离的投影点将被丢弃以避免矩阵病态
        """
        self.X_train = None
        self.N = 0
        self.use_projection = use_projection
        self.min_proj_distance = min_proj_distance
        
        # 梯度插值使用的所有基点（原始 + 有效投影点）
        self.X_cf_base = None
        self.N_cf = 0  # 梯度插值的基点数量
        
        # 旋度自由基底系数
        self.c_cf = None 
        self.b_cf = None
        
        # 标量残差基底系数
        self.c_sc = None
        self.b_sc = None

        # 标量残差插值使用的基点（原始 + 有效投影点）
        self.X_sc_base = None
        self.N_sc = 0

        self.trained = False

    def _filter_projected_points(self, original_points, projected_points, gradients):
        """
        过滤投影点：只保留与所有已有点（原始+已接受的投影点）距离足够远的投影点。
        返回有效投影点的索引。
        """
        N = original_points.shape[0]
        valid_mask = np.ones(N, dtype=bool)
        
        for i in range(N):
            # 检查投影点到所有原始点的距离
            dists_to_orig = np.linalg.norm(projected_points[i] - original_points, axis=1)
            if np.min(dists_to_orig) < self.min_proj_distance:
                valid_mask[i] = False
                continue
            
            # 检查投影点到已接受的其他投影点的距离
            if np.any(valid_mask[:i]):
                accepted_proj = projected_points[:i][valid_mask[:i]]
                dists_to_proj = np.linalg.norm(projected_points[i] - accepted_proj, axis=1)
                if len(dists_to_proj) > 0 and np.min(dists_to_proj) < self.min_proj_distance:
                    valid_mask[i] = False
        
        return valid_mask

    def fit(self, sdf_points, sdf_values, sdf_gradients):
        """
        根据给定的 SDF 点集、距离值和梯度进行插值拟合。
        """
        self.X_train = np.asarray(sdf_points)
        sdf_values = np.asarray(sdf_values)
        sdf_gradients = np.asarray(sdf_gradients)
        self.N = self.X_train.shape[0]

        # ---------------------------------------------------------
        # 阶段 1：利用旋度自由 RBF 拟合 SDF 的梯度场 (产生初始势能场)
        # ---------------------------------------------------------
        if self.use_projection:
            # 计算投影点：P_proj = P - S * d
            projected_points = self.X_train - sdf_values[:, np.newaxis] * sdf_gradients
            
            # 过滤掉太近的投影点
            valid_mask = self._filter_projected_points(self.X_train, projected_points, sdf_gradients)
            valid_proj = projected_points[valid_mask]
            valid_proj_grads = sdf_gradients[valid_mask]
            
            # 合并原始点和有效投影点
            self.X_cf_base = np.vstack([self.X_train, valid_proj])
            self.N_cf = self.X_cf_base.shape[0]
            all_grads = np.vstack([sdf_gradients, valid_proj_grads])
            
            print(f"  Projection: {valid_mask.sum()}/{self.N} projected points accepted, "
                  f"total CF base points: {self.N_cf}")
        else:
            self.X_cf_base = self.X_train
            self.N_cf = self.N
            all_grads = sdf_gradients
        
        A_cf, P_cf = self._build_cf_matrix_general(self.X_cf_base)
        
        # 组装全局矩阵
        M_cf = np.block([
            [A_cf, P_cf],
            [P_cf.T, np.zeros((5, 5))]
        ])
        
        # 展平所有梯度数据并组装 RHS
        u = np.zeros(2 * self.N_cf)
        u[0::2] = all_grads[:, 0]
        u[1::2] = all_grads[:, 1]
        RHS_cf = np.concatenate((u, np.zeros(5)))
        
        # 求解 CF 系数
        sol_cf = np.linalg.solve(M_cf, RHS_cf)
        self.c_cf = sol_cf[:2 * self.N_cf].reshape((self.N_cf, 2))
        self.b_cf = sol_cf[2 * self.N_cf:]

        # ---------------------------------------------------------
        # 阶段 2：计算初始势能场的偏差，并用标量 RBF 拟合残差
        # 同样使用原始点 + 有效投影点（投影点 SDF=0）
        # ---------------------------------------------------------
        if self.use_projection:
            self.X_sc_base = np.vstack([self.X_train, valid_proj])
            self.N_sc = self.X_sc_base.shape[0]
            # 投影点的真实 SDF 值为 0
            all_sdf_for_sc = np.concatenate([sdf_values, np.zeros(len(valid_proj))])
        else:
            self.X_sc_base = self.X_train
            self.N_sc = self.N
            all_sdf_for_sc = sdf_values

        initial_potential = self._eval_cf_potential(self.X_sc_base)
        residual = all_sdf_for_sc - initial_potential
        
        A_sc, P_sc = self._build_scalar_matrix(self.X_sc_base)
        
        M_sc = np.block([
            [A_sc, P_sc],
            [P_sc.T, np.zeros((3, 3))]
        ])
        
        RHS_sc = np.concatenate((residual, np.zeros(3)))
        
        sol_sc = np.linalg.solve(M_sc, RHS_sc)
        self.c_sc = sol_sc[:self.N_sc]
        self.b_sc = sol_sc[self.N_sc:]
        self.trained = True

    def predict(self, query_points):
        """
        对给定的新位置点集计算预测的 SDF 值。
        """
        query_points = np.asarray(query_points)
        # 阶段 1：基础势能预测
        V_base = self._eval_cf_potential(query_points)
        # 阶段 2：残差补偿预测
        V_correction = self._eval_scalar_potential(query_points)
        
        return V_base + V_correction

    # =========================================================
    # 内部向量化数学计算方法
    # =========================================================
    def _build_cf_matrix_general(self, X):
        """
        构建旋度自由梯度插值矩阵，适用于任意数量的基点。
        基于 PHS 核 phi(r) = r^4 log r 的 Hessian 矩阵。
        """
        N_pts = X.shape[0]
        dx = X[:, 0:1] - X[:, 0:1].T
        dy = X[:, 1:2] - X[:, 1:2].T
        r2 = dx**2 + dy**2
        r = np.sqrt(r2)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            log_r = np.log(r)
            log_r[r == 0] = 0.0
            
        term1 = 8 * log_r + 6
        term2 = r2 * (4 * log_r + 1)
        
        p11 = term1 * dx**2 + term2
        p12 = term1 * dx * dy
        p22 = term1 * dy**2 + term2
        
        p11[r == 0] = 0; p12[r == 0] = 0; p22[r == 0] = 0
        
        A = np.zeros((2 * N_pts, 2 * N_pts))
        A[0::2, 0::2] = p11
        A[0::2, 1::2] = p12
        A[1::2, 0::2] = p12
        A[1::2, 1::2] = p22
        
        P = np.zeros((2 * N_pts, 5))
        P[0::2, 0] = 1
        P[1::2, 1] = 1
        P[0::2, 2] = X[:, 1]
        P[1::2, 2] = X[:, 0]
        P[0::2, 3] = X[:, 0]
        P[1::2, 4] = X[:, 1]
        
        return A, P

    def _build_cf_matrix(self, X):
        """兼容旧接口"""
        return self._build_cf_matrix_general(X)

    def _build_scalar_matrix(self, X):
        # 2D TPS 核 \phi(r) = r^2 \log r，多项式空间 P_1（3个基）
        N_pts = X.shape[0]
        dx = X[:, 0:1] - X[:, 0:1].T
        dy = X[:, 1:2] - X[:, 1:2].T
        r2 = dx**2 + dy**2
        r = np.sqrt(r2)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            log_r = np.log(r)
            log_r[r == 0] = 0.0
            
        A = r2 * log_r  # r^2 log r
        A[r == 0] = 0
        
        # P_1 多项式空间: [1, x, y]（3 个基）
        P = np.ones((N_pts, 3))
        P[:, 1] = X[:, 0]
        P[:, 2] = X[:, 1]
        return A, P

    def _eval_cf_potential(self, X_query):
        # 使用梯度插值的基点（可能包含投影点）
        base = self.X_cf_base
        dx = X_query[:, 0:1] - base[:, 0]  # 形状: (M, N_cf)
        dy = X_query[:, 1:2] - base[:, 1]
        r2 = dx**2 + dy**2
        r = np.sqrt(r2)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            log_r = np.log(r)
            log_r[r == 0] = 0.0
            
        coef = r2 * (4 * log_r + 1)
        coef[r == 0] = 0.0
        
        # -\nabla^T \phi_2 * c_j
        term = coef * (dx * self.c_cf[:, 0] + dy * self.c_cf[:, 1])
        V = np.sum(term, axis=1)
        
        x, y = X_query[:, 0], X_query[:, 1]
        b = self.b_cf
        V += b[0]*x + b[1]*y + b[2]*x*y + 0.5*b[3]*x**2 + 0.5*b[4]*y**2
        return V

    def _eval_scalar_potential(self, X_query):
        # 使用标量残差插值的基点（可能包含投影点）
        base = self.X_sc_base
        dx = X_query[:, 0:1] - base[:, 0]
        dy = X_query[:, 1:2] - base[:, 1]
        r2 = dx**2 + dy**2
        r = np.sqrt(r2)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            log_r = np.log(r)
            log_r[r == 0] = 0.0
            
        phi = r2 * log_r  # r^2 log r
        phi[r == 0] = 0.0
        
        V = np.dot(phi, self.c_sc)
        x, y = X_query[:, 0], X_query[:, 1]
        # P_1: b_0 + b_1*x + b_2*y
        V += self.b_sc[0] + self.b_sc[1]*x + self.b_sc[2]*y
        return V
    
class Interpolator:
    """
    A Duchon interpolator to fit and predict values based on input signed distance data.
    """
    def __init__(self, kernel):
        """
        Initialize the Duchon interpolation object with a specified radial basis function kernel.
        Parameters
        ----------
        kernel : str
            The type of radial basis function to use. Supported options are:
            - 'thin_plate': Uses r^2 * log(r) as the kernel function. Good for 2D
            - Other values: Defaults to cubic kernel (r^3). Good for 3D
        Attributes
        ----------
        points : None
            Will store the interpolation points (initialized as None)
        values : None
            Will store the values at interpolation points (initialized as None)
        alpha : None
            Will store the interpolation coefficients (initialized as None)
        beta : None
            Will store the gradient coefficients (initialized as None)
        p : None
            Will store polynomial coefficients (initialized as None)
        q : None
            Will store additional coefficients (initialized as None)
        kernel : callable
            The radial basis function used for interpolation
        """
        self.points = None
        self.values = None
        self.alpha = None
        self.beta = None
        self.p = None
        self.q = None
        self.kernel_type = kernel
        self.trained = False
        if kernel == 'thin_plate':
            self.kernel = lambda r: r**2 * np.log(r + 1e-10)  # Adding a small value to avoid log(0)
        else:
            self.kernel = lambda r: r**3  # Default to cubic kernel

    def fit(self, points, values, gradients=None):
        """
        Fit the interpolator with given points and their corresponding values.

        Parameters:
        points (np.ndarray): An array of shape (n_samples, m_dimensions) representing the input points.
        values (np.ndarray): An array of shape (n_samples,) representing the values at the input points.
        gradients (np.ndarray, optional): An array of shape (n_samples, m_dimensions) representing the gradients at the input points. If provided, the interpolator will also fit to these gradients. Default is None.
        """
        self.points = points
        self.values = values
        if gradients is not None:
            self.alpha, self.beta, self.p, self.q = self._compute_coefficients_with_gradients(points, values, gradients)
        else:
            self.alpha, self.p, self.q = self._compute_coefficients(points, values)
        self.trained = True
    
    def _compute_coefficients_with_gradients(self, points, values, gradients):
        """
        Compute Hermite RBF interpolation coefficients that fit both values and gradients.

        Builds and solves the extended system:
            [ Φ       -D_0     -D_1    | P_val  ] [α  ]   [f  ]
            [ D_0^T    H_00     H_01   | P_gx  ] [β_x] = [g_x]
            [ D_1^T    H_01^T   H_11   | P_gy  ] [β_y]   [g_y]
            [ P_val^T  P_gx^T   P_gy^T | 0     ] [poly]   [0  ]

        Parameters:
        points (np.ndarray): (n, d) input points.
        values (np.ndarray): (n,) values.
        gradients (np.ndarray): (n, d) gradients.

        Returns:
        tuple: (alpha (n,), beta (n,d), p (d,), q (scalar))
        """
        n = points.shape[0]
        d = points.shape[1]

        # Pairwise differences and distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (N, N, d)
        r2 = np.sum(diff ** 2, axis=2)  # (N, N)
        dist = np.sqrt(r2)              # (N, N)

        # ---- Kernel matrix Φ (N×N): value-vs-value ----
        with np.errstate(divide='ignore', invalid='ignore'):
            if self.kernel_type == 'thin_plate':
                log_r = np.log(dist)
                log_r[dist == 0] = 0.0
                Phi = r2 * log_r
            else:
                Phi = dist ** 3
        np.fill_diagonal(Phi, 0.0)

        # ---- First derivative coefficient (shared) ----
        with np.errstate(divide='ignore', invalid='ignore'):
            if self.kernel_type == 'thin_plate':
                coeff1 = 2 * log_r + 1       # (N, N)
            else:
                coeff1 = 3 * dist            # (N, N)
        coeff1[dist == 0] = 0.0

        # D_k[i,j] = coeff1 * (x_i_k - x_j_k)  (first derivative w.r.t. eval point)
        D = [coeff1 * diff[:, :, k] for k in range(d)]

        # ---- Second mixed derivative factor ----
        with np.errstate(divide='ignore', invalid='ignore'):
            if self.kernel_type == 'thin_plate':
                safe_r2 = r2.copy(); safe_r2[dist == 0] = 1.0
                factor = 2.0 / safe_r2       # 2 d_l d_k / r^2 term
            else:
                safe_dist = dist.copy(); safe_dist[dist == 0] = 1.0
                factor = 3.0 / safe_dist     # 3 d_l d_k / r term

        # H_lk[i,j] = ∂²φ / (∂x_i_l ∂x_j_k) = -[ factor*d_l*d_k + coeff1*δ_{lk} ]
        H = {}
        for l in range(d):
            for k in range(d):
                Hlk = -(factor * diff[:, :, l] * diff[:, :, k])
                if l == k:
                    Hlk = Hlk - coeff1
                Hlk[dist == 0] = 0.0
                H[(l, k)] = Hlk

        # ---- Assemble full system  (size = n*(1+d) + (d+1)) ----
        size = n * (1 + d) + (d + 1)
        M = np.zeros((size, size))
        rhs = np.zeros(size)

        # Block (value, α): Φ
        M[:n, :n] = Phi
        # Block (value, β_k): -D_k  (derivative w.r.t. center = -D_k)
        for k in range(d):
            M[:n, n + k * n: n + (k + 1) * n] = -D[k]
        # Block (grad_l, α): D_l  (by symmetry equals (-D_l)^T = D_l since D_l is antisymmetric)
        for l in range(d):
            M[n + l * n: n + (l + 1) * n, :n] = D[l]
        # Block (grad_l, β_k): H_lk
        for l in range(d):
            for k in range(d):
                M[n + l * n: n + (l + 1) * n, n + k * n: n + (k + 1) * n] = H[(l, k)]

        # Polynomial columns / orthogonality rows
        ps = n * (1 + d)  # poly start index
        # Value rows: [1, x_0, x_1, ...]
        M[:n, ps] = 1.0
        for k in range(d):
            M[:n, ps + 1 + k] = points[:, k]
        # Grad_l rows: [0, ..., 1 at position l, ...]
        for l in range(d):
            M[n + l * n: n + (l + 1) * n, ps + 1 + l] = 1.0
        # Orthogonality rows (transpose of polynomial columns)
        M[ps, :n] = 1.0
        for k in range(d):
            M[ps + 1 + k, :n] = points[:, k]
            M[ps + 1 + k, n + k * n: n + (k + 1) * n] = 1.0

        # RHS
        rhs[:n] = values
        for l in range(d):
            rhs[n + l * n: n + (l + 1) * n] = gradients[:, l]

        # Solve
        coefficients, _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)

        alpha = coefficients[:n]
        beta = np.zeros((n, d))
        for k in range(d):
            beta[:, k] = coefficients[n + k * n: n + (k + 1) * n]
        q = coefficients[ps]
        p = coefficients[ps + 1: ps + 1 + d]
        return alpha, beta, p, q

    def _compute_coefficients(self, points, values):
        """
        Compute the coefficients for the Duchon interpolation based on the input points and values.
        Parameters:
        points (np.ndarray): An array of shape (n_samples, m_dimensions) representing the input points.
        values (np.ndarray): An array of shape (n_samples,) representing the values at the input points.
        
        Returns:
        tuple: A tuple containing the coefficients for the radial basis functions and polynomial terms.
        """
        # construct the interpolation matrix
        n_samples = points.shape[0]
        m_dimensions = points.shape[1]
        K = np.zeros((n_samples + m_dimensions + 1, n_samples + m_dimensions + 1))
        distances = cdist(points, points, metric='euclidean')
        K_block = self.kernel(distances)
        np.fill_diagonal(K_block, 0)  # kernel(0) = 0
        K[:n_samples, :n_samples] = K_block
        # Add polynomial terms for Duchon interpolation
        P = np.ones((n_samples, m_dimensions + 1))
        P[:, :-1] = points
        K[:n_samples, n_samples:] = P
        K[n_samples:, :n_samples] = P.T
        y = np.zeros(n_samples + m_dimensions + 1)
        y[:n_samples] = values
        # Solve for coefficients (use lstsq to handle near-singular matrices)
        coefficients, _, _, _ = np.linalg.lstsq(K, y, rcond=None)
        return coefficients[:n_samples], coefficients[n_samples:-1], coefficients[-1]

    def predict(self, x_new : np.ndarray):
        """
        Predict values at new input points using the fitted interpolator. Duchon interpolation multiplies all basis
        functions by a coefficient term. The basis functions are radial basis functions that depend on the distance
        between points.

        Parameters:
        x_new (np.ndarray): An array of shape (m_samples, dimensions) representing the new input points.

        Returns:
        np.ndarray: An array of shape (m_samples,) representing the predicted values at the new points.
        """
        distances = cdist(x_new, self.points, metric='euclidean')
        r = self.kernel(distances)  # Apply kernel to all distances at once
        result = r @ self.alpha + x_new @ self.p + self.q

        if self.beta is not None:
            # Add gradient basis contributions: Σ_j Σ_k β_{jk} · ψ_{jk}(x)
            # ψ_{jk}(x) = ∂φ/∂(x_j)_k = -coeff1 · (x_k - x_{j,k})
            diff = x_new[:, np.newaxis, :] - self.points[np.newaxis, :, :]  # (M, N, d)
            with np.errstate(divide='ignore', invalid='ignore'):
                if self.kernel_type == 'thin_plate':
                    log_r = np.log(distances)
                    log_r[distances == 0] = 0.0
                    coeff1 = 2 * log_r + 1
                else:
                    coeff1 = 3 * distances
            coeff1[distances == 0] = 0.0
            for k in range(self.points.shape[1]):
                psi_k = -coeff1 * diff[:, :, k]  # (M, N)
                result += psi_k @ self.beta[:, k]

        return result
    
    def predict_gradient(self, x_new):
        """
        Predict gradients at new input points using the fitted interpolator.

        Parameters:
        x_new (np.ndarray): An array of shape (m_samples, dimensions) representing the new input points.

        Returns:
        np.ndarray: An array of shape (m_samples, dimensions) representing the predicted gradients at the new points.
        """
        # Vectorized: compute all diffs and distances at once
        # diff shape: (m_samples, n_samples, dimensions)
        diff = x_new[:, np.newaxis, :] - self.points[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2, keepdims=True)  # (m_samples, n_samples, 1)
        dist = np.maximum(dist, 1e-10)  # Avoid division by zero
        if self.kernel_type == 'thin_plate':
            kernel_deriv = (2 * np.log(dist) + 1) * diff  # (m, n, d)
        else:
            kernel_deriv = 3 * dist * diff  # (m, n, d)
        # Weighted sum over training points: alpha[j] * kernel_deriv[i,j,:] → gradients[i,:]
        gradients = np.einsum('j,ijd->id', self.alpha, kernel_deriv)
        gradients += self.p  # Add polynomial term gradient

        if self.beta is not None:
            # Add Hessian contribution: Σ_j Σ_k β_{jk} · ∂²φ/(∂x_l ∂(x_j)_k)
            dist_sq = dist[..., 0]  # (m, n) squeeze keepdims
            dist_flat = dist_sq  # already (m, n)
            diff_nd = diff  # (m, n, d)
            with np.errstate(divide='ignore', invalid='ignore'):
                if self.kernel_type == 'thin_plate':
                    safe_r2 = (dist_flat ** 2).copy()
                    safe_r2[dist_flat < 1e-10] = 1.0
                    factor = 2.0 / safe_r2
                    coeff1 = 2 * np.log(dist_flat) + 1
                else:
                    safe_dist = dist_flat.copy()
                    safe_dist[dist_flat < 1e-10] = 1.0
                    factor = 3.0 / safe_dist
                    coeff1 = 3 * dist_flat
            coeff1[dist_flat < 1e-10] = 0.0
            factor[dist_flat < 1e-10] = 0.0

            d_dim = self.points.shape[1]
            for l in range(d_dim):
                grad_l_beta = np.zeros(x_new.shape[0])
                for k in range(d_dim):
                    Hlk = -(factor * diff_nd[:, :, l] * diff_nd[:, :, k])
                    if l == k:
                        Hlk = Hlk - coeff1
                    Hlk[dist_flat < 1e-10] = 0.0
                    grad_l_beta += Hlk @ self.beta[:, k]
                gradients[:, l] += grad_l_beta

        return gradients
    
    def sample_best_gradient(self, x_new, sdf, num_samples=10):
        """
        Sample a series of gradients around the input point x_new and project the x_new to the surface along that gradient.
        Predict the SDF values and return the gradient that has the smallest absolute predicted value.

        Parameters:
        x_new (np.ndarray): An array of shape (dimensions,) representing the new input point.
        sdf (float): The signed distance value at the input point.
        num_samples (int): The number of directions to sample around the input point. Default is 10.

        Returns:
        np.ndarray: An array of shape (dimensions,) representing the best gradient direction for the input point.
        """
        # divide the unit circle into num_samples directions uniformly in 2D
        angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
        directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        directions = np.vstack([directions, self.predict_gradient(x_new.reshape(1, -1))])  # Add the predicted gradient as an additional direction
        samples = x_new - sdf * directions
        predictions = self.predict(samples)
        if sdf > 0:
            best_idx = np.argmin(predictions)  # For points outside the surface, we want to minimize the predicted SDF
        else:
            best_idx = np.argmax(predictions)  # For points inside the surface, we want to maximize the predicted SDF
        return directions[best_idx] / np.linalg.norm(directions[best_idx])  # Normalize the best direction
    
    def sample_best_gradients(self, x_new: np.ndarray, sdf: np.ndarray, num_samples=50):
        """
        Batch version of sample_best_gradient for multiple input points. Sample a series of gradients around each input point and project the points to the surface along those gradients.
        Predict the SDF values and return the gradients that have the smallest absolute predicted value for each point.
        
        Parameters:
        x_new (np.ndarray): An array of shape (batch_size, dimensions) representing the new input points.
        sdf (np.ndarray): An array of shape (batch_size,) representing the signed distance values at the input points.
        num_samples (int): The number of directions to sample around each input point. Default is 10.

        Returns:
        np.ndarray: An array of shape (batch_size, dimensions) representing the best gradient directions for each input point.
        """
        # divide the unit circle into num_samples directions uniformly in 2D
        angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
        directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # Shape (num_samples, 2)
        batch_size = x_new.shape[0]
        # directions = np.vstack([directions, self.predict_gradient(x_new).reshape(-1, 2)])  # Add the predicted gradients as additional directions, shape (num_samples + batch_size, 2)
        samples = x_new[:, np.newaxis, :] - sdf[:, np.newaxis, np.newaxis] * directions[np.newaxis, :, :]  # Shape (batch_size, num_samples + batch_size, 2)
        predictions = self.predict(samples.reshape(-1, 2)).reshape(batch_size, -1)  # Shape (batch_size, num_samples + batch_size)
        best_indices = np.where(sdf.ravel() > 0, np.argmin(predictions, axis=1), np.argmax(predictions, axis=1))  # Shape (batch_size,)
        best_directions = directions[best_indices]  # Shape (batch_size, 2)
        best_directions /= np.linalg.norm(best_directions, axis=1, keepdims=True)  # Normalize the best directions
        return best_directions
