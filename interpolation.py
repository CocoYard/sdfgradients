import numpy as np
from scipy.spatial.distance import cdist

class CurlFree_Interpolator:
    """
    基于旋度自由 RBF (Curl-Free RBF) 和标量薄板样条 (Thin-Plate Spline) 
    的 SDF 全局插值器。
    """
    def __init__(self):
        self.X_train = None
        self.N = 0
        
        # 旋度自由基底系数
        self.c_cf = None 
        self.b_cf = None
        
        # 标量残差基底系数
        self.c_sc = None
        self.b_sc = None

        self.trained = False

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
        A_cf, P_cf = self._build_cf_matrix(self.X_train)
        
        # 组装全局矩阵
        M_cf = np.block([
            [A_cf, P_cf],
            [P_cf.T, np.zeros((5, 5))]
        ])
        
        # 展平梯度数据并组装 RHS
        u = np.zeros(2 * self.N)
        u[0::2] = sdf_gradients[:, 0]
        u[1::2] = sdf_gradients[:, 1]
        RHS_cf = np.concatenate((u, np.zeros(5)))
        
        # 求解 CF 系数
        sol_cf = np.linalg.solve(M_cf, RHS_cf)
        self.c_cf = sol_cf[:2 * self.N].reshape((self.N, 2))
        self.b_cf = sol_cf[2 * self.N:]

        # ---------------------------------------------------------
        # 阶段 2：计算初始势能场的偏差，并用标量 RBF 拟合残差
        # ---------------------------------------------------------
        # 获取采样点上当前的势能估计值
        initial_potential = self._eval_cf_potential(self.X_train)
        
        # 计算我们需要补偿的残差 (目标值 - 当前值)
        residual = sdf_values - initial_potential
        
        # 构建薄板样条 (Thin-Plate Spline) 标量矩阵
        A_sc, P_sc = self._build_scalar_matrix(self.X_train)
        
        M_sc = np.block([
            [A_sc, P_sc],
            [P_sc.T, np.zeros((3, 3))]
        ])
        
        RHS_sc = np.concatenate((residual, np.zeros(3)))
        
        # 求解标量 RBF 系数
        sol_sc = np.linalg.solve(M_sc, RHS_sc)
        self.c_sc = sol_sc[:self.N]
        self.b_sc = sol_sc[self.N:]
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
    def _build_cf_matrix(self, X):
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
        
        # 处理 r=0 的奇点
        p11[r == 0] = 0; p12[r == 0] = 0; p22[r == 0] = 0
        
        A = np.zeros((2 * self.N, 2 * self.N))
        A[0::2, 0::2] = p11
        A[0::2, 1::2] = p12
        A[1::2, 0::2] = p12
        A[1::2, 1::2] = p22
        
        P = np.zeros((2 * self.N, 5))
        P[0::2, 0] = 1
        P[1::2, 1] = 1
        P[0::2, 2] = X[:, 1]
        P[1::2, 2] = X[:, 0]
        P[0::2, 3] = X[:, 0]
        P[1::2, 4] = X[:, 1]
        
        return A, P

    def _build_scalar_matrix(self, X):
        # 2D 标量薄板样条 \phi(r) = r^2 \log r
        dx = X[:, 0:1] - X[:, 0:1].T
        dy = X[:, 1:2] - X[:, 1:2].T
        r2 = dx**2 + dy**2
        r = np.sqrt(r2)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            log_r = np.log(r)
            log_r[r == 0] = 0.0
            
        A = r2 * log_r
        A[r == 0] = 0
        
        P = np.ones((self.N, 3))
        P[:, 1] = X[:, 0]
        P[:, 2] = X[:, 1]
        return A, P

    def _eval_cf_potential(self, X_query):
        # 利用广播快速计算所有 query_points 到所有 train_points 的距离向量
        dx = X_query[:, 0:1] - self.X_train[:, 0] # 形状: (M, N)
        dy = X_query[:, 1:2] - self.X_train[:, 1] # <- 改成 1:2
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
        dx = X_query[:, 0:1] - self.X_train[:, 0]
        dy = X_query[:, 1:2] - self.X_train[:, 1]
        r2 = dx**2 + dy**2
        r = np.sqrt(r2)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            log_r = np.log(r)
            log_r[r == 0] = 0.0
            
        phi = r2 * log_r
        phi[r == 0] = 0.0
        
        V = np.dot(phi, self.c_sc)
        x, y = X_query[:, 0], X_query[:, 1]
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
        self.p = None
        self.q = None
        self.kernel_type = kernel
        self.trained = False
        if kernel == 'thin_plate':
            self.kernel = lambda r: r**2 * np.log(r + 1e-10)  # Adding a small value to avoid log(0)
        else:
            self.kernel = lambda r: r**3  # Default to cubic kernel

    def fit(self, points, values):
        """
        Fit the interpolator with given points and their corresponding values.

        Parameters:
        points (np.ndarray): An array of shape (n_samples, m_dimensions) representing the input points.
        values (np.ndarray): An array of shape (n_samples,) representing the values at the input points.
        """
        self.points = points
        self.values = values
        self.alpha, self.p, self.q = self._compute_coefficients(points, values)
        self.trained = True

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
        n_samples = self.points.shape[0]
        m_samples = x_new.shape[0]
        distances = cdist(x_new, self.points, metric='euclidean')
        r = self.kernel(distances)  # Apply kernel to all distances at once
        return r @ self.alpha + x_new @ self.p + self.q
    
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
