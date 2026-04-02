import numpy as np
from scipy.spatial.distance import cdist

class Interpolator:
    """
    A Duchon interpolator to fit and predict values based on input signed distance data.
    """
    def __init__(self, kernel='thin_plate', use_projection=False):
        """
        Initialize the Duchon interpolation object with a specified radial basis function kernel.
        Parameters
        ----------
        kernel : str
            The type of radial basis function to use. Supported options are:
            - 'thin_plate': Uses r^2 * log(r) as the kernel function. Good for 2D
            - Other values: Defaults to cubic kernel (r^3). Good for 3D
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
            self.kernel = lambda r: r**2 * np.log(r + 1e-10)
        else:
            self.kernel = lambda r: r**3

    def fit(self, points, values, gradients=None, mask=None, force_recompute=False, hermite_interp=False, distance_threshold=0.2):
        """
        Fit the interpolator with given points and their corresponding values.

        Parameters:
        points (np.ndarray): An array of shape (n_samples, m_dimensions) representing the input points.
        values (np.ndarray): An array of shape (n_samples,) representing the values at the input points.
        gradients (np.ndarray, optional): An array of shape (n_samples, m_dimensions) representing the gradients at 
            the input points. If they are provided and hermite_interp is True, the interpolator will be of Hermite form. Otherwise,
            also use projections based on gradients to do interpolation. Default is None.
        mask (np.ndarray, optional): A boolean array of shape (n_samples,) indicating which points to use for fitting. Because
            some projection points are in the invisible region.
        force_recompute (bool, optional): If True, forces the interpolator to refit even if it has already been trained. Default is False.
        hermite_interp (bool, optional): If True and gradients are provided, uses Hermite interpolation. Default is False.
        distance_threshold (float, optional): If the number of points exceeds 5000, only keep points with absolute values less than this
            threshold to reduce time and memory usage. Default is 0.2.
        """
        if self.trained and not force_recompute:
            print(f"Interpolator is already trained. Use force_recompute=True to refit {points.shape[0]} points.")
            return
        if gradients is not None and not hermite_interp:
            if mask is None:
                projections = points - values[:, np.newaxis] * gradients
            else:
                projections = points[mask] - values[mask, np.newaxis] * gradients[mask]
            self.points = np.vstack([points, projections])
            self.values = np.concatenate([values, np.zeros(len(projections))])
        elif hermite_interp:
            assert gradients is not None, "Hermite interpolation requires gradients to be provided."
            self.points = points
            self.values = values
            self.alpha, self.beta, self.p, self.q = self._compute_coefficients_with_gradients(points, values, gradients)
        else:
            self.points = points
            self.values = values
        if len(self.values) > 5000:
            close_mask = np.abs(self.values) < distance_threshold
            self.points = self.points[close_mask]
            self.values = self.values[close_mask]
            print(f"Warning: too many points, only keeping {len(self.values)} points with abs(value) < {distance_threshold} for fitting.")
        if not hermite_interp:
            self.alpha, self.p, self.q = self._compute_coefficients(self.points, self.values)
        self.trained = True
    
    def _compute_coefficients_with_gradients(self, points, values, gradients):
        """
        Compute Hermite RBF interpolation coefficients that fit both values and gradients.
        """
        n = points.shape[0]
        d = points.shape[1]

        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        r2 = np.sum(diff ** 2, axis=2)
        dist = np.sqrt(r2)

        with np.errstate(divide='ignore', invalid='ignore'):
            if self.kernel_type == 'thin_plate':
                log_r = np.log(dist)
                log_r[dist == 0] = 0.0
                Phi = r2 * log_r
            else:
                Phi = dist ** 3
        np.fill_diagonal(Phi, 0.0)

        with np.errstate(divide='ignore', invalid='ignore'):
            if self.kernel_type == 'thin_plate':
                coeff1 = 2 * log_r + 1
            else:
                coeff1 = 3 * dist
        coeff1[dist == 0] = 0.0

        D = [coeff1 * diff[:, :, k] for k in range(d)]

        with np.errstate(divide='ignore', invalid='ignore'):
            if self.kernel_type == 'thin_plate':
                safe_r2 = r2.copy(); safe_r2[dist == 0] = 1.0
                factor = 2.0 / safe_r2
            else:
                safe_dist = dist.copy(); safe_dist[dist == 0] = 1.0
                factor = 3.0 / safe_dist

        H = {}
        for l in range(d):
            for k in range(d):
                Hlk = -(factor * diff[:, :, l] * diff[:, :, k])
                if l == k:
                    Hlk = Hlk - coeff1
                Hlk[dist == 0] = 0.0
                H[(l, k)] = Hlk

        size = n * (1 + d) + (d + 1)
        M = np.zeros((size, size))
        rhs = np.zeros(size)

        M[:n, :n] = Phi
        for k in range(d):
            M[:n, n + k * n: n + (k + 1) * n] = -D[k]
        for l in range(d):
            M[n + l * n: n + (l + 1) * n, :n] = D[l]
        for l in range(d):
            for k in range(d):
                M[n + l * n: n + (l + 1) * n, n + k * n: n + (k + 1) * n] = H[(l, k)]

        ps = n * (1 + d)
        M[:n, ps] = 1.0
        for k in range(d):
            M[:n, ps + 1 + k] = points[:, k]
        for l in range(d):
            M[n + l * n: n + (l + 1) * n, ps + 1 + l] = 1.0
        M[ps, :n] = 1.0
        for k in range(d):
            M[ps + 1 + k, :n] = points[:, k]
            M[ps + 1 + k, n + k * n: n + (k + 1) * n] = 1.0

        rhs[:n] = values
        for l in range(d):
            rhs[n + l * n: n + (l + 1) * n] = gradients[:, l]

        coefficients, _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)

        alpha = coefficients[:n]
        beta = np.zeros((n, d))
        for k in range(d):
            beta[:, k] = coefficients[n + k * n: n + (k + 1) * n]
        q = coefficients[ps]
        p = coefficients[ps + 1: ps + 1 + d]
        return alpha, beta, p, q

    def _compute_coefficients(self, points, values):
        n_samples = points.shape[0]
        m_dimensions = points.shape[1]
        K = np.zeros((n_samples + m_dimensions + 1, n_samples + m_dimensions + 1))
        distances = cdist(points, points, metric='euclidean')
        K_block = self.kernel(distances)
        np.fill_diagonal(K_block, 0)
        K[:n_samples, :n_samples] = K_block
        P = np.ones((n_samples, m_dimensions + 1))
        P[:, :-1] = points
        K[:n_samples, n_samples:] = P
        K[n_samples:, :n_samples] = P.T
        y = np.zeros(n_samples + m_dimensions + 1)
        y[:n_samples] = values
        coefficients, _, _, _ = np.linalg.lstsq(K, y, rcond=None)
        return coefficients[:n_samples], coefficients[n_samples:-1], coefficients[-1]

    def predict(self, x_new: np.ndarray, chunk_size: int = 500):
        """
        Predict values at new input points using the fitted interpolator.
        chunk_size controls how many query points are processed at once to limit memory usage.
        """
        if len(x_new) > chunk_size:
            return np.concatenate([self.predict(x_new[s:s + chunk_size], chunk_size)
                                   for s in range(0, len(x_new), chunk_size)])
        distances = cdist(x_new, self.points, metric='euclidean')
        r = self.kernel(distances)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            result = r @ self.alpha + x_new @ self.p + self.q
        assert not np.any(np.isinf(result)), "predict: result contains inf"
        if self.beta is not None:
            diff = x_new[:, np.newaxis, :] - self.points[np.newaxis, :, :]
            with np.errstate(divide='ignore', invalid='ignore'):
                if self.kernel_type == 'thin_plate':
                    log_r = np.log(distances)
                    log_r[distances == 0] = 0.0
                    coeff1 = 2 * log_r + 1
                else:
                    coeff1 = 3 * distances
            coeff1[distances == 0] = 0.0
            for k in range(self.points.shape[1]):
                psi_k = -coeff1 * diff[:, :, k]
                result += psi_k @ self.beta[:, k]

        return result
    
    def extract_zero_level_set(self, bounds, resolution=256):
        """
        Extract zero level set contours (Marching Squares for 2D / Marching Cubes for 3D)
        
        Parameters
        ----------
        bounds : tuple
            2D: ((xmin, xmax), (ymin, ymax))
            3D: ((xmin, xmax), (ymin, ymax), (zmin, zmax))
        resolution : int
            Grid resolution per axis (default 256)
        
        Returns
        -------
        2D: list of np.ndarray
            List of polyline vertex arrays, each with shape (M, 2)
        3D: tuple (vertices, faces)
            Vertex array (N, 3) and triangle face index array (M, 3)
        """
        # Detect dimensionality
        is_3d = len(bounds) == 3
        
        if is_3d:
            return self._extract_zero_level_set_3d(bounds, resolution)
        else:
            return self._extract_zero_level_set_2d(bounds, resolution)
    
    def _extract_zero_level_set_2d(self, bounds, resolution=256):
        """
        2D Marching Squares algorithm for extracting iso-contours.
        """
        (xmin, xmax), (ymin, ymax) = bounds
        xs = np.linspace(xmin, xmax, resolution)
        ys = np.linspace(ymin, ymax, resolution)
        X, Y = np.meshgrid(xs, ys)
        grid_pts = np.column_stack([X.ravel(), Y.ravel()])
        Z = self.predict(grid_pts).reshape(resolution, resolution)
        
        dx = (xmax - xmin) / (resolution - 1)
        dy = (ymax - ymin) / (resolution - 1)
        
        # --- Marching Squares: collect edge segments ---
        # For each 2x2 cell, classify corners by sign and interpolate crossings
        signs = (Z >= 0).astype(np.int8)  # 0 = negative, 1 = positive
        
        # Cell corner indices: TL=top-left (row i, col j), TR, BL, BR
        # Row i goes downward (y increases), col j goes rightward (x increases)
        TL = signs[:-1, :-1]
        TR = signs[:-1, 1:]
        BL = signs[1:, :-1]
        BR = signs[1:, 1:]
        case = TL * 8 + TR * 4 + BR * 2 + BL  # 0-15 case index
        
        # Values at corners
        vTL = Z[:-1, :-1]
        vTR = Z[:-1, 1:]
        vBL = Z[1:, :-1]
        vBR = Z[1:, 1:]
        
        # Precompute interpolation fractions on each edge (0->1 along the edge)
        def lerp_frac(va, vb):
            denom = va - vb
            denom[denom == 0] = 1e-30
            return va / denom
        
        frac_top = lerp_frac(vTL, vTR)     # top edge: TL -> TR
        frac_bottom = lerp_frac(vBL, vBR)  # bottom edge: BL -> BR
        frac_left = lerp_frac(vTL, vBL)    # left edge: TL -> BL
        frac_right = lerp_frac(vTR, vBR)   # right edge: TR -> BR
        
        nr, nc = case.shape  # (resolution-1, resolution-1)
        
        # Edge midpoint coordinates for a cell (row_i, col_j):
        #   top:    (xmin + (j + frac)*dx, ymin + i*dy)
        #   bottom: (xmin + (j + frac)*dx, ymin + (i+1)*dy)
        #   left:   (xmin + j*dx,          ymin + (i + frac)*dy)
        #   right:  (xmin + (j+1)*dx,      ymin + (i + frac)*dy)
        row_idx, col_idx = np.mgrid[:nr, :nc]
        
        top_x = xmin + (col_idx + frac_top) * dx
        top_y = ymin + row_idx * dy
        bottom_x = xmin + (col_idx + frac_bottom) * dx
        bottom_y = ymin + (row_idx + 1) * dy
        left_x = xmin + col_idx * dx
        left_y = ymin + (row_idx + frac_left) * dy
        right_x = xmin + (col_idx + 1) * dx
        right_y = ymin + (row_idx + frac_right) * dy
        
        # Edge point arrays: shape (nr, nc, 2)
        pt_top = np.stack([top_x, top_y], axis=-1)
        pt_bottom = np.stack([bottom_x, bottom_y], axis=-1)
        pt_left = np.stack([left_x, left_y], axis=-1)
        pt_right = np.stack([right_x, right_y], axis=-1)
        
        # Marching squares lookup: case -> list of (edge_a, edge_b) line segments
        # Edges: 0=top, 1=right, 2=bottom, 3=left
        _edge_table = {
            0: [], 1: [(2, 3)], 2: [(1, 2)], 3: [(1, 3)],
            4: [(0, 1)], 5: [(0, 3), (1, 2)], 6: [(0, 2)], 7: [(0, 3)],
            8: [(0, 3)], 9: [(0, 2)], 10: [(0, 1), (2, 3)], 11: [(0, 1)],
            12: [(1, 3)], 13: [(1, 2)], 14: [(2, 3)], 15: [],
        }
        edge_pts = [pt_top, pt_right, pt_bottom, pt_left]
        
        # Collect all segments
        segments = []
        for i in range(nr):
            for j in range(nc):
                c = case[i, j]
                for ea, eb in _edge_table[c]:
                    pa = edge_pts[ea][i, j]
                    pb = edge_pts[eb][i, j]
                    segments.append((tuple(pa), tuple(pb)))
        
        if not segments:
            return []
        
        # --- Chain segments into polylines ---
        from collections import defaultdict
        
        adj = defaultdict(list)
        for idx, (a, b) in enumerate(segments):
            adj[a].append((b, idx))
            adj[b].append((a, idx))
        
        used = [False] * len(segments)
        contours = []
        
        for start_idx in range(len(segments)):
            if used[start_idx]:
                continue
            used[start_idx] = True
            a, b = segments[start_idx]
            chain = [a, b]
            
            # Extend forward from b
            cur = b
            while True:
                found = False
                for nxt, seg_idx in adj[cur]:
                    if not used[seg_idx]:
                        used[seg_idx] = True
                        chain.append(nxt)
                        cur = nxt
                        found = True
                        break
                if not found:
                    break
            
            # Extend backward from a
            cur = a
            while True:
                found = False
                for nxt, seg_idx in adj[cur]:
                    if not used[seg_idx]:
                        used[seg_idx] = True
                        chain.insert(0, nxt)
                        cur = nxt
                        found = True
                        break
                if not found:
                    break
            
            contours.append(np.array(chain))

        self.contour_resolution = resolution
        self.zero_contours = contours
        return contours
    
    def _extract_zero_level_set_3d(self, bounds, grid_resolution=64):
        """
        3D Marching Cubes algorithm for extracting zero level set (triangle mesh).
        
        Uses scikit-image's built-in implementation.
        
        Parameters
        ----------
        bounds : ((xmin, xmax), (ymin, ymax), (zmin, zmax))
            3D bounding box
        grid_resolution : int
            Grid resolution per axis (default 64 for 3D to avoid memory explosion)
        
        Returns
        -------
        vertices : (N, 3) ndarray
            Triangle mesh vertex coordinates
        faces : (M, 3) ndarray
            Triangle face vertex indices (0-indexed)
        """
        try:
            from skimage.measure import marching_cubes
        except ImportError:
            raise ImportError("scikit-image required for 3D Marching Cubes support: pip install scikit-image")
        lx = np.linspace(bounds[0][0], bounds[0][1], grid_resolution)
        ly = np.linspace(bounds[1][0], bounds[1][1], grid_resolution)
        lz = np.linspace(bounds[2][0], bounds[2][1], grid_resolution)
        xx, yy, zz = np.meshgrid(lx, ly, lz, indexing='ij')
        pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        grid_values = self.predict(pts).reshape(grid_resolution, grid_resolution, grid_resolution)
        # Extract isosurface at value 0 using marching cubes
        from skimage import measure
        sp = ((lx[-1]-lx[0])/(grid_resolution-1), (ly[-1]-ly[0])/(grid_resolution-1), (lz[-1]-lz[0])/(grid_resolution-1))
        verts, faces, normals, values = measure.marching_cubes(grid_values, level=0.0, spacing=sp)
        verts += np.array([lx[0], ly[0], lz[0]])
        
        return verts, faces

    def predict_gradient(self, x_new):
        """
        Predict gradients at new input points using the fitted interpolator.
        """
        diff = x_new[:, np.newaxis, :] - self.points[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2, keepdims=True)
        dist = np.maximum(dist, 1e-10)
        if self.kernel_type == 'thin_plate':
            kernel_deriv = (2 * np.log(dist) + 1) * diff
        else:
            kernel_deriv = 3 * dist * diff
        gradients = np.einsum('j,ijd->id', self.alpha, kernel_deriv)
        gradients += self.p

        if self.beta is not None:
            dist_sq = dist[..., 0]
            dist_flat = dist_sq
            diff_nd = diff
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
    
    def sample_gradient_by_alignment(self, x_new, sdf, num_coarse=24, tol=1e-6, visible_arcs=None, initial_guess=None):
        """
        Find the best gradient direction by coarse sweep + bounded scalar optimization.

        First performs a coarse uniform sweep over the unit circle to locate a
        promising angular region, then refines with scipy.optimize.minimize_scalar
        for high precision.

        Parameters:
        x_new (np.ndarray): Shape (dimensions,) — the query point.
        sdf (float): The signed distance value at the query point.
        num_coarse (int): Number of uniformly spaced directions in the coarse sweep. Default 24.
        tol (float): Absolute tolerance for the angular refinement. Default 1e-6.
        visible_arcs (list of tuples): Optional list of (start_angle, end_angle) arcs in radians that are visible.
        initial_guess (float): Initial guess for the optimal angle. Default None. If provided, the coarse sweep will skip.
        The bounded scalar optimization will be centered around this angle.

        Returns:
        np.ndarray: Shape (dimensions,) — the best gradient direction (unit vector).
        """
        from scipy.optimize import minimize_scalar
        sign = 1.0 if sdf > 0 else -1.0
        def objective(angle):
            direction = np.array([np.cos(angle), np.sin(angle)])
            sample = (x_new + np.abs(sdf) * direction).reshape(1, -1)
            pred_grad = self.predict_gradient(sample)[0]
            pred_grad /= np.linalg.norm(pred_grad) + 1e-10
            return sign * direction @ pred_grad
        if initial_guess is not None:
            best_angle = initial_guess
        else:
            # Coarse sweep (uniform angles)
            if visible_arcs is not None:
                angles = []
                for arc in visible_arcs:
                    num_cuts = (arc[1] - arc[0]) / (2 * np.pi) * num_coarse if arc[1] > arc[0] else (arc[1] - arc[0] + 2 * np.pi) / (2 * np.pi) * num_coarse
                    arc_angles = np.linspace(arc[0], arc[1], int(np.ceil(num_cuts)), endpoint=False)
                    angles.extend(arc_angles)
                angles = np.array(angles)
            else:
                angles = np.linspace(0, 2 * np.pi, num_coarse, endpoint=False)

            all_dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
            samples = x_new + np.abs(sdf) * all_dirs
            pred_grads = self.predict_gradient(samples)
            pred_grads = pred_grads / np.linalg.norm(pred_grads, axis=1, keepdims=True)
            best_idx = np.argmin(np.sum(sign * all_dirs * pred_grads, axis=1))
            best_angle = angles[best_idx]
        # Refine with bounded scalar optimization around the best coarse angle
        delta = np.pi / num_coarse
        bounds = [best_angle - delta, best_angle + delta]
        result = minimize_scalar(objective,
                                 bounds=bounds,
                                 method='bounded',
                                 options={'xatol': tol})
        best_angle = result.x
        direction = np.array([np.cos(best_angle), np.sin(best_angle)])
        return -direction * sign
    
    def sample_best_gradient(self, x_new, sdf, num_coarse=24, tol=1e-6, initial_guess=None):
        """
        Find the best gradient direction by coarse sweep + bounded scalar optimization.

        First performs a coarse uniform sweep over the unit circle to locate a
        promising angular region, then refines with scipy.optimize.minimize_scalar
        for high precision.

        Parameters:
        x_new (np.ndarray): Shape (dimensions,) — the query point.
        sdf (float): The signed distance value at the query point.
        num_coarse (int): Number of uniformly spaced directions in the coarse sweep. Default 24.
        tol (float): Absolute tolerance for the angular refinement. Default 1e-6.
        initial_guess (float): Initial guess for the optimal angle. Default None. If provided, the coarse sweep will skip.
        The bounded scalar optimization will be centered around this angle.

        Returns:
        np.ndarray: Shape (dimensions,) — the best gradient direction (unit vector).
        """
        from scipy.optimize import minimize_scalar
        if initial_guess is not None:
            best_angle = initial_guess
        else:
            sign = 1.0 if sdf > 0 else -1.0

            def objective(angle):
                direction = np.array([np.cos(angle), np.sin(angle)])
                sample = (x_new - sdf * direction).reshape(1, -1)
                return sign * self.predict(sample)[0]

            # Coarse sweep (uniform angles)
            angles = np.linspace(0, 2 * np.pi, num_coarse, endpoint=False)

            all_dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
            samples = x_new - sdf * all_dirs
            preds = self.predict(samples)
            obj_vals = sign * preds
            best_idx = np.argmin(obj_vals)
            best_angle = angles[best_idx]
        # Refine with bounded scalar optimization around the best coarse angle
        delta = np.pi / num_coarse
        result = minimize_scalar(objective,
                                 bounds=(best_angle - delta, best_angle + delta),
                                 method='bounded',
                                 options={'xatol': tol})
        best_angle = result.x
        direction = np.array([np.cos(best_angle), np.sin(best_angle)])
        return direction

    def sample_best_gradients(self, x_new, sdf, num_coarse=24,
                              refine_steps=4, num_refine=12, initial_guess=None, chunk_size=200):
        """
        Batch version: find best gradient directions via coarse sweep + iterative
        narrowing refinement (similar to binary search on the angle).
        Phase 1 — coarse uniform sweep over the full circle to locate the best
        angular region per point.
        Phase 2 — repeatedly zoom into a smaller interval around the current best
        angle and re-evaluate, shrinking the search window each iteration.
        
        Parameters:
        -----------
        x_new (np.ndarray): Shape (batch_size, dimensions) — input points.
        sdf (np.ndarray): Shape (batch_size,) — signed distance values.
        num_coarse (int): Directions in the initial coarse sweep. Default 24.
        refine_steps (int): Number of zoom-in refinement iterations. Default 4.
        num_refine (int): Directions evaluated per refinement step. Default 12.
        initial_guess (np.ndarray): Initial guesses for the optimal angles. Default None. If provided, the coarse sweep will skip and the first refinement will 
        be centered around these angles.

        Returns:
        --------
        np.ndarray: Shape (batch_size, dimensions) — best gradient directions (unit vectors).
        """
        if x_new.shape[1] == 3:
            return self._sample_best_gradients_3d(x_new, sdf, num_coarse,
                                                 refine_steps, num_refine, initial_guess, chunk_size)
        else:
            return self._sample_best_gradients_2d(x_new, sdf, num_coarse,
                                                 refine_steps, num_refine, initial_guess, chunk_size)
        
    def _sample_best_gradients_2d(self, x_new, sdf, num_coarse=24,
                                  refine_steps=4, num_refine=12, initial_guess=None, chunk_size=200):
        """
        Batch version: find best gradient directions via coarse sweep + iterative
        narrowing refinement (similar to binary search on the angle).

        Phase 1 — coarse uniform sweep over the full circle to locate the best
        angular region per point.
        Phase 2 — repeatedly zoom into a smaller interval around the current best
        angle and re-evaluate, shrinking the search window each iteration.

        Parameters:
        x_new (np.ndarray): Shape (batch_size, dimensions) — input points.
        sdf (np.ndarray): Shape (batch_size,) — signed distance values.
        num_coarse (int): Directions in the initial coarse sweep. Default 24.
        refine_steps (int): Number of zoom-in refinement iterations. Default 4.
        num_refine (int): Directions evaluated per refinement step. Default 12.
        initial_guess (np.ndarray): Initial guesses for the optimal angles. Default None. 
        If provided, the coarse sweep will skip and the first refinement will 
        be centered around these angles.

        Returns:
        np.ndarray: Shape (batch_size, dimensions) — best gradient directions (unit vectors).
        """
        batch_size = x_new.shape[0]
        sdf_flat = sdf.ravel()
        sign = np.where(sdf_flat > 0, 1.0, -1.0)  # (batch,)

        # --- Phase 1: coarse sweep ---
        if initial_guess is not None:
            best_angles = initial_guess
        else:
            angles = np.linspace(0, 2 * np.pi, num_coarse, endpoint=False)  # (C,)
            dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)       # (C, 2)

            # samples: (batch, C, 2)
            samples = x_new[:, None, :] - sdf_flat[:, None, None] * dirs[None, :, :]
            preds = self.predict(samples.reshape(-1, 2)).reshape(batch_size, num_coarse)
            obj = preds * sign[:, None]
            best_idx = np.argmin(obj, axis=1)           # (batch,)
            best_angles = angles[best_idx]              # (batch,)

        # --- Phase 2: iterative refinement ---
        half_range = np.pi / num_coarse  # initial half-width of search window

        for _ in range(refine_steps):
            offsets = np.linspace(-1.0, 1.0, num_refine)               # (R,)
            local_angles = best_angles[:, None] + half_range * offsets  # (batch, R)

            cos_a = np.cos(local_angles)  # (batch, R)
            sin_a = np.sin(local_angles)
            # samples: (batch, R, 2)
            samples = x_new[:, None, :] - sdf_flat[:, None, None] * np.stack([cos_a, sin_a], axis=2)
            preds = self.predict(samples.reshape(-1, 2)).reshape(batch_size, num_refine)
            obj = preds * sign[:, None]
            best_local = np.argmin(obj, axis=1)
            best_angles = local_angles[np.arange(batch_size), best_local]

            # Shrink window: next half_range = one spacing of current grid
            half_range = 2.0 * half_range / (num_refine - 1)

        best_dirs = np.stack([np.cos(best_angles), np.sin(best_angles)], axis=1)
        best_dirs /= np.linalg.norm(best_dirs, axis=1, keepdims=True)
        return best_dirs
    

    def _sample_best_gradients_3d(self, x_new, sdf, num_coarse=64,
                                  refine_steps=4, num_refine=16, initial_guess=None,
                                  chunk_size=200):
        """
        3D version: find best gradient directions via coarse sphere sweep + iterative
        cone refinement.

        Phase 1 — coarse uniform sweep over the full sphere using Fibonacci sampling.
        Phase 2 — repeatedly zoom into a cone around the current best direction,
                   sampling a Fibonacci disk within the cone.

        Parameters:
        x_new (np.ndarray): Shape (batch_size, 3) — input points.
        sdf (np.ndarray): Shape (batch_size,) — signed distance values.
        num_coarse (int): Directions in the initial coarse sweep. Default 64.
        refine_steps (int): Number of cone-narrowing refinement iterations. Default 4.
        num_refine (int): Directions evaluated per refinement step. Default 16.
        initial_guess (np.ndarray): Initial guesses (batch_size, 3) unit vectors. Default None.

        Returns:
        np.ndarray: Shape (batch_size, 3) — best gradient directions (unit vectors).
        """
        batch_size = x_new.shape[0]
        sdf_flat = sdf.ravel()
        sign = np.where(sdf_flat > 0, 1.0, -1.0)  # (batch,)

        def fibonacci_sphere(n):
            """Uniformly distributed directions on unit sphere."""
            golden = (1 + np.sqrt(5)) / 2
            i = np.arange(n, dtype=float)
            theta = 2 * np.pi * i / golden
            phi = np.arccos(1 - 2 * (i + 0.5) / n)
            return np.stack([np.sin(phi)*np.cos(theta),
                             np.sin(phi)*np.sin(theta),
                             np.cos(phi)], axis=1)  # (n, 3)

        def tangent_frame(d):
            """Build orthonormal tangent vectors t1, t2 perpendicular to d (batch, 3)."""
            ref = np.where(np.abs(d[:, 0:1]) < 0.9,
                           np.broadcast_to([1., 0., 0.], d.shape).copy(),
                           np.broadcast_to([0., 1., 0.], d.shape).copy())
            t1 = np.cross(d, ref)
            t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
            t2 = np.cross(d, t1)
            return t1, t2  # each (batch, 3)

        def cone_dirs(best_dirs, half_angle, n):
            """Sample n directions in a cone of half_angle around each best_dir."""
            golden = (1 + np.sqrt(5)) / 2
            i = np.arange(n, dtype=float)
            r = half_angle * np.sqrt((i + 0.5) / n)   # radial angle within cone (n,)
            alpha = 2 * np.pi * i / golden             # azimuth within cone (n,)
            t1, t2 = tangent_frame(best_dirs)          # (batch, 3)
            # world_dirs: (batch, n, 3)
            dirs = (np.cos(r)[None, :, None] * best_dirs[:, None, :]
                    + np.sin(r)[None, :, None] * (np.cos(alpha)[None, :, None] * t1[:, None, :]
                                                  + np.sin(alpha)[None, :, None] * t2[:, None, :]))
            return dirs  # (batch, n, 3)

        # --- Phase 1: coarse sweep ---
        if initial_guess is not None:
            best_dirs = initial_guess / np.linalg.norm(initial_guess, axis=1, keepdims=True)
        else:
            dirs = fibonacci_sphere(num_coarse)  # (C, 3)
            samples = x_new[:, None, :] - sdf_flat[:, None, None] * dirs[None, :, :]
            preds = self.predict(samples.reshape(-1, 3), chunk_size).reshape(batch_size, num_coarse)
            obj = preds * sign[:, None]
            best_idx = np.argmin(obj, axis=1)
            best_dirs = dirs[best_idx]  # (batch, 3)

        # --- Phase 2: iterative cone refinement ---
        half_angle = np.pi / np.sqrt(num_coarse)

        for _ in range(refine_steps):
            dirs_r = cone_dirs(best_dirs, half_angle, num_refine)  # (batch, R, 3)
            samples = x_new[:, None, :] - sdf_flat[:, None, None] * dirs_r
            preds = self.predict(samples.reshape(-1, 3), chunk_size).reshape(batch_size, num_refine)
            obj = preds * sign[:, None]
            best_local = np.argmin(obj, axis=1)
            best_dirs = dirs_r[np.arange(batch_size), best_local]
            best_dirs /= np.linalg.norm(best_dirs, axis=1, keepdims=True)
            half_angle /= np.sqrt(num_refine)
        return best_dirs

class CurlFree_Interpolator(Interpolator):
    """
    A global SDF interpolator based on Curl-Free RBF and Thin-Plate Spline (TPS).
    Inherits from Interpolator.
    
    When use_projection=True, projects original sample points along gradient directions
    to the surface vicinity, using them as additional gradient constraint points
    to enhance interpolation accuracy.
    """
    def __init__(self, min_proj_distance=1e-8):
        super().__init__(kernel='thin_plate')
        self.X_train = None
        self.N = 0
        self.min_proj_distance = min_proj_distance
        
        # --- Curl-Free (gradient field) related interpolation variables ---
        self.X_cf_base = None  # Base points for gradient interpolation (original + filtered projected points)
        self.N_cf = 0          # Total number of CF base points (N_cf)

        self.c_cf = None       # RBF weight coefficients for curl-free field (shape: N_cf x 2)
        self.b_cf = None       # Polynomial coefficients for curl-free potential (shape: 5)

        # --- Scalar/residual field related interpolation variables ---
        self.X_sc_base = None  # Base points for scalar residual interpolation
        self.N_sc = 0          # Total number of scalar base points (N_sc)

        self.c_sc = None       # RBF weight coefficients for scalar field (shape: N_sc)
        self.b_sc = None       # Polynomial coefficients for scalar potential (shape: 3)

        self.trained = False

    def _filter_projected_points(self, original_points, projected_points, gradients):
        """
        Filter projected points: keep only those far enough from all existing points
        (original + already accepted projected points).
        """
        N = original_points.shape[0]
        valid_mask = np.ones(N, dtype=bool)
        
        for i in range(N):
            dists_to_orig = np.linalg.norm(projected_points[i] - original_points, axis=1)
            if np.min(dists_to_orig) < self.min_proj_distance:
                valid_mask[i] = False
                continue
            
            if np.any(valid_mask[:i]):
                accepted_proj = projected_points[:i][valid_mask[:i]]
                dists_to_proj = np.linalg.norm(projected_points[i] - accepted_proj, axis=1)
                if len(dists_to_proj) > 0 and np.min(dists_to_proj) < self.min_proj_distance:
                    valid_mask[i] = False
        
        return valid_mask

    def fit(self, sdf_points, sdf_values, sdf_gradients, force_recompute=False, use_projection=True):
        """
        Fit the interpolator with given SDF points, distance values, and gradients.
        """
        if self.trained and not force_recompute:
            print(f"Interpolator is already trained. Use force_recompute=True to refit {sdf_points.shape[0]} points.")
            return
        self.X_train = np.asarray(sdf_points)
        sdf_values = np.asarray(sdf_values)
        sdf_gradients = np.asarray(sdf_gradients)
        self.N = self.X_train.shape[0]

        if use_projection:
            projected_points = self.X_train - sdf_values[:, np.newaxis] * sdf_gradients
            valid_mask = self._filter_projected_points(self.X_train, projected_points, sdf_gradients)
            valid_proj = projected_points[valid_mask]
            valid_proj_grads = sdf_gradients[valid_mask]
            
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
        
        M_cf = np.block([
            [A_cf, P_cf],
            [P_cf.T, np.zeros((5, 5))]
        ])
        
        u = np.zeros(2 * self.N_cf)
        u[0::2] = all_grads[:, 0]
        u[1::2] = all_grads[:, 1]
        RHS_cf = np.concatenate((u, np.zeros(5)))
        
        sol_cf = np.linalg.solve(M_cf, RHS_cf)
        self.c_cf = sol_cf[:2 * self.N_cf].reshape((self.N_cf, 2))
        self.b_cf = sol_cf[2 * self.N_cf:]

        if use_projection:
            self.X_sc_base = np.vstack([self.X_train, valid_proj])
            self.N_sc = self.X_sc_base.shape[0]
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
        Predict SDF values at given query point locations.
        """
        query_points = np.asarray(query_points)
        V_base = self._eval_cf_potential(query_points)
        V_correction = self._eval_scalar_potential(query_points)
        return V_base + V_correction

    def predict_gradient(self, x_new, use_gradient_field=False):
        """
        Compute predicted gradients at given query point locations.
        """
        if use_gradient_field:
            return self._eval_cf_flow(x_new)
        else:
            # # Numerical gradient approximation
            # eps = 1e-5
            # grad = np.zeros_like(x_new)
            # for k in range(2):
            #     shift = np.zeros(2)
            #     shift[k] = eps
            #     grad[:, k] = (self.predict(x_new + shift) - self.predict(x_new - shift)) / (2 * eps)
            # return grad / np.linalg.norm(grad, axis=1, keepdims=True)

            term1 = self._eval_cf_flow(x_new)
            term2 = self._eval_scalar_gradient(x_new)
            total_grad = term1 + term2
            return total_grad / np.linalg.norm(total_grad, axis=1, keepdims=True)

    # =========================================================
    # Internal vectorized mathematical computation methods
    # =========================================================
    def _build_cf_matrix_general(self, X):
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
        return self._build_cf_matrix_general(X)

    def _build_scalar_matrix(self, X):
        N_pts = X.shape[0]
        dx = X[:, 0:1] - X[:, 0:1].T
        dy = X[:, 1:2] - X[:, 1:2].T
        r2 = dx**2 + dy**2
        r = np.sqrt(r2)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            log_r = np.log(r)
            log_r[r == 0] = 0.0
            
        A = r2 * log_r
        A[r == 0] = 0
        
        P = np.ones((N_pts, 3))
        P[:, 1] = X[:, 0]
        P[:, 2] = X[:, 1]
        return A, P
    
    def _eval_cf_flow(self, X_query):
        """
        Compute curl-free vector field values at query points.
        
        Vector field = RBF Hessian matrix components dot-product with weight coefficients + polynomial gradient
        
        Parameters
        ----------
        X_query : (M, 2) array
            Query point coordinates
        
        Returns
        -------
        flow : (M, 2) array
            Curl-free vector field values (gradient vectors) at each query point
        """
        base = self.X_cf_base
        dx = X_query[:, 0:1] - base[:, 0]  # (M, N_cf)
        dy = X_query[:, 1:2] - base[:, 1]  # (M, N_cf)
        r2 = dx**2 + dy**2
        r = np.sqrt(r2)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            log_r = np.log(r)
            log_r[r == 0] = 0.0
        
        # --- RBF Hessian matrix components ---
        # H = r^2(4*log_r + 1)*I + (8*log_r + 6)*(dx*dx, dx*dy; dy*dx, dy*dy)
        term1 = 8 * log_r + 6
        term2 = r2 * (4 * log_r + 1)
        
        # H_xx = (8*log_r + 6)*dx^2 + r^2(4*log_r + 1)
        # H_yy = (8*log_r + 6)*dy^2 + r^2(4*log_r + 1)
        # H_xy = (8*log_r + 6)*dx*dy
        H_xx = term1 * dx**2 + term2  # (M, N_cf)
        H_yy = term1 * dy**2 + term2  # (M, N_cf)
        H_xy = term1 * dx * dy        # (M, N_cf)
        
        H_xx[r == 0] = 0
        H_yy[r == 0] = 0
        H_xy[r == 0] = 0
        
        # --- Weight coefficient vector c_cf has shape (N_cf, 2) ---
        # Flow x-component = sum_j (H_xx[j] * c_cf[j, 0] + H_xy[j] * c_cf[j, 1])
        # Flow y-component = sum_j (H_xy[j] * c_cf[j, 0] + H_yy[j] * c_cf[j, 1])
        flow_x = np.sum(H_xx * self.c_cf[:, 0] + H_xy * self.c_cf[:, 1], axis=1)  # (M,)
        flow_y = np.sum(H_xy * self.c_cf[:, 0] + H_yy * self.c_cf[:, 1], axis=1)  # (M,)
        
        # --- Polynomial gradient contribution ---
        # P(x, y) = b[0]*x + b[1]*y + b[2]*x*y + 0.5*b[3]*x^2 + 0.5*b[4]*y^2
        # dP/dx = b[0] + b[2]*y + b[3]*x
        # dP/dy = b[1] + b[2]*x + b[4]*y
        x = X_query[:, 0]
        y = X_query[:, 1]
        b = self.b_cf
        
        flow_x += b[0] + b[2]*y + b[3]*x
        flow_y += b[1] + b[2]*x + b[4]*y
        
        return np.column_stack([flow_x, flow_y])  # (M, 2)

    def _eval_scalar_gradient(self, X_query):
        """
        Compute scalar potential field (TPS) gradient at query points.
        
        Scalar field gradient = TPS RBF gradient dot-product with weight coefficients + polynomial gradient
        
        Parameters
        ----------
        X_query : (M, 2) array
            Query point coordinates
        
        Returns
        -------
        grad : (M, 2) array
            Scalar potential field gradient vectors at each query point
        """
        base = self.X_sc_base
        dx = X_query[:, 0:1] - base[:, 0]  # (M, N_sc)
        dy = X_query[:, 1:2] - base[:, 1]  # (M, N_sc)
        r2 = dx**2 + dy**2
        r = np.sqrt(r2)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            log_r = np.log(r)
            log_r[r == 0] = 0.0
        
        # --- TPS RBF gradient components ---
        # φ(r) = r^2 * log(r)
        # ∇φ = 2(2*log(r) + 1) * (Δx, Δy)
        coef = 2 * (2 * log_r + 1)  # (M, N_sc)
        coef[r == 0] = 0.0
        
        # Gradient x-component = sum_j (2(2*log_r + 1) * Δx[j] * c_sc[j])
        # Gradient y-component = sum_j (2(2*log_r + 1) * Δy[j] * c_sc[j])
        grad_x = np.sum(coef * dx * self.c_sc, axis=1)  # (M,)
        grad_y = np.sum(coef * dy * self.c_sc, axis=1)  # (M,)
        
        # --- Polynomial gradient contribution ---
        # P(x, y) = b_sc[0] + b_sc[1]*x + b_sc[2]*y
        # dP/dx = b_sc[1]
        # dP/dy = b_sc[2]
        grad_x += self.b_sc[1]
        grad_y += self.b_sc[2]
        
        return np.column_stack([grad_x, grad_y])  # (M, 2)

    def _eval_cf_potential(self, X_query):
        base = self.X_cf_base
        dx = X_query[:, 0:1] - base[:, 0]
        dy = X_query[:, 1:2] - base[:, 1]
        r2 = dx**2 + dy**2
        r = np.sqrt(r2)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            log_r = np.log(r)
            log_r[r == 0] = 0.0
            
        coef = r2 * (4 * log_r + 1)
        coef[r == 0] = 0.0
        
        term = coef * (dx * self.c_cf[:, 0] + dy * self.c_cf[:, 1])
        V = np.sum(term, axis=1)
        
        x, y = X_query[:, 0], X_query[:, 1]
        b = self.b_cf
        V += b[0]*x + b[1]*y + b[2]*x*y + 0.5*b[3]*x**2 + 0.5*b[4]*y**2
        return V

    def _eval_scalar_potential(self, X_query):
        base = self.X_sc_base
        dx = X_query[:, 0:1] - base[:, 0]
        dy = X_query[:, 1:2] - base[:, 1]
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
