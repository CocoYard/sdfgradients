from collections import deque
import numpy as np
from scipy.spatial.distance import cdist
from abc import ABC, abstractmethod
from scipy.spatial import KDTree
import time

DEBUG_TIME = True
DEBUG_COVER = False

class Interpolator(ABC):
    """
    Abstract base class for interpolation. Subclasses must implement fit(), predict(),
    predict_gradients(), and sample_best_gradients().
    """
    @abstractmethod
    def fit(self, points, values, gradients=None, **kwargs):
        pass

    @abstractmethod
    def predict(self, x_new, chunk_size=500):
        pass

    @abstractmethod
    def predict_gradients(self, x_new, chunk_size=500):
        pass

    def extract_zero_level_set(self, bounds, resolution=256):
        """
        Extract zero level set contours (Marching Squares for 2D / Marching Cubes for 3D)
        """
        if len(bounds) == 3:
            return self._extract_zero_level_set_3d(bounds, resolution)
        else:
            return self._extract_zero_level_set_2d(bounds, resolution)

    def _extract_zero_level_set_2d(self, bounds, resolution=256):
        (xmin, xmax), (ymin, ymax) = bounds
        xs = np.linspace(xmin, xmax, resolution)
        ys = np.linspace(ymin, ymax, resolution)
        X, Y = np.meshgrid(xs, ys)
        grid_pts = np.column_stack([X.ravel(), Y.ravel()])
        Z = self.predict(grid_pts).reshape(resolution, resolution)

        dx = (xmax - xmin) / (resolution - 1)
        dy = (ymax - ymin) / (resolution - 1)

        signs = (Z >= 0).astype(np.int8)
        TL = signs[:-1, :-1]; TR = signs[:-1, 1:]
        BL = signs[1:, :-1];  BR = signs[1:, 1:]
        case = TL * 8 + TR * 4 + BR * 2 + BL

        vTL = Z[:-1, :-1]; vTR = Z[:-1, 1:]
        vBL = Z[1:, :-1];  vBR = Z[1:, 1:]

        def lerp_frac(va, vb):
            denom = va - vb
            denom[denom == 0] = 1e-30
            return va / denom

        frac_top = lerp_frac(vTL, vTR)
        frac_bottom = lerp_frac(vBL, vBR)
        frac_left = lerp_frac(vTL, vBL)
        frac_right = lerp_frac(vTR, vBR)

        nr, nc = case.shape
        row_idx, col_idx = np.mgrid[:nr, :nc]

        pt_top = np.stack([xmin + (col_idx + frac_top) * dx, ymin + row_idx * dy], axis=-1)
        pt_bottom = np.stack([xmin + (col_idx + frac_bottom) * dx, ymin + (row_idx + 1) * dy], axis=-1)
        pt_left = np.stack([xmin + col_idx * dx, ymin + (row_idx + frac_left) * dy], axis=-1)
        pt_right = np.stack([xmin + (col_idx + 1) * dx, ymin + (row_idx + frac_right) * dy], axis=-1)

        _edge_table = {
            0: [], 1: [(2, 3)], 2: [(1, 2)], 3: [(1, 3)],
            4: [(0, 1)], 5: [(0, 3), (1, 2)], 6: [(0, 2)], 7: [(0, 3)],
            8: [(0, 3)], 9: [(0, 2)], 10: [(0, 1), (2, 3)], 11: [(0, 1)],
            12: [(1, 3)], 13: [(1, 2)], 14: [(2, 3)], 15: [],
        }
        edge_pts = [pt_top, pt_right, pt_bottom, pt_left]

        segments = []
        for i in range(nr):
            for j in range(nc):
                c = case[i, j]
                for ea, eb in _edge_table[c]:
                    segments.append((tuple(edge_pts[ea][i, j]), tuple(edge_pts[eb][i, j])))

        if not segments:
            return []

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
            for end, insert_fn in [(b, chain.append), (a, lambda x: chain.insert(0, x))]:
                cur = end
                while True:
                    found = False
                    for nxt, seg_idx in adj[cur]:
                        if not used[seg_idx]:
                            used[seg_idx] = True
                            insert_fn(nxt)
                            cur = nxt
                            found = True
                            break
                    if not found:
                        break
            contours.append(np.array(chain))

        self.contour_resolution = resolution
        self.zero_contours = contours
        return contours

    def _extract_zero_level_set_3d(self, bounds, grid_resolution=64, use_dual_contouring=False):
        if use_dual_contouring:
            from occupancy_dual_contouring import occupancy_dual_contouring
            import torch
            if torch.cuda.is_available():
                device = "cuda:1"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            print(f"[dual contouring] using device: {device}")
            odc = occupancy_dual_contouring(device);
            def predict_wrapper(xyz: torch.Tensor) -> torch.Tensor:
                pts = xyz.cpu().numpy().astype(np.float64)
                sdf = self.predict(pts)
                return torch.from_numpy(sdf.astype(np.float32)).to(xyz.device)
            min_bound = np.array([b[0] for b in bounds], dtype=np.float32)
            max_bound = np.array([b[1] for b in bounds], dtype=np.float32)
            vertices, triangles = odc.extract_mesh(predict_wrapper, min_coord=min_bound, max_coord=max_bound, num_grid=grid_resolution);
            return vertices.cpu().numpy(), triangles.cpu().numpy()
        try:
            from skimage.measure import marching_cubes
        except ImportError:
            raise ImportError("scikit-image required for 3D Marching Cubes: pip install scikit-image")
        lx = np.linspace(bounds[0][0], bounds[0][1], grid_resolution)
        ly = np.linspace(bounds[1][0], bounds[1][1], grid_resolution)
        lz = np.linspace(bounds[2][0], bounds[2][1], grid_resolution)
        xx, yy, zz = np.meshgrid(lx, ly, lz, indexing='ij')
        pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        grid_values = self.predict(pts).reshape(grid_resolution, grid_resolution, grid_resolution)
        from skimage import measure
        sp = ((lx[-1]-lx[0])/(grid_resolution-1), (ly[-1]-ly[0])/(grid_resolution-1), (lz[-1]-lz[0])/(grid_resolution-1))
        verts, faces, normals, values = measure.marching_cubes(grid_values, level=0.0, spacing=sp)
        verts += np.array([lx[0], ly[0], lz[0]])
        return verts, faces

    def sample_best_gradients(self, x_new, sdf, num_coarse=24, given_samples=None,
                              refine_steps=4, num_refine=12, initial_guess=None, chunk_size=200):
        """
        Find best gradient directions via coarse sweep + iterative refinement.
        Works for any subclass since it only calls self.predict().
        """
        if given_samples is not None:
            batch_size = x_new.shape[0]
            num_coarse = given_samples.shape[1]
            sign = np.where(sdf > 0, 1.0, -1.0)
            preds = self.predict(given_samples.reshape(-1, x_new.shape[1])).reshape(batch_size, num_coarse)
            obj = preds * sign[:, None]
            best_idx = np.argmin(obj, axis=1)
            best_grads = x_new - given_samples[np.arange(batch_size), best_idx]
            best_grads /= sdf[:, None] + 1e-10
            initial_guess = best_grads

        if x_new.shape[1] == 3:
            return self._sample_best_gradients_3d(x_new, sdf, num_coarse,
                                                  refine_steps, num_refine, initial_guess, chunk_size)
        else:
            return self._sample_best_gradients_2d(x_new, sdf, num_coarse,
                                                  refine_steps, num_refine, initial_guess, chunk_size)

    def _sample_best_gradients_2d(self, x_new, sdf, num_coarse=24,
                                  refine_steps=4, num_refine=12, initial_guess=None, chunk_size=200):
        batch_size = x_new.shape[0]
        sdf_flat = sdf.ravel()
        sign = np.where(sdf_flat > 0, 1.0, -1.0)

        if initial_guess is not None:
            best_angles = initial_guess
        else:
            angles = np.linspace(0, 2 * np.pi, num_coarse, endpoint=False)
            dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
            samples = x_new[:, None, :] - sdf_flat[:, None, None] * dirs[None, :, :]
            preds = self.predict(samples.reshape(-1, 2)).reshape(batch_size, num_coarse)
            obj = preds * sign[:, None]
            best_idx = np.argmin(obj, axis=1)
            best_angles = angles[best_idx]

        half_range = np.pi / num_coarse
        for _ in range(refine_steps):
            offsets = np.linspace(-1.0, 1.0, num_refine)
            local_angles = best_angles[:, None] + half_range * offsets
            cos_a = np.cos(local_angles)
            sin_a = np.sin(local_angles)
            samples = x_new[:, None, :] - sdf_flat[:, None, None] * np.stack([cos_a, sin_a], axis=2)
            preds = self.predict(samples.reshape(-1, 2)).reshape(batch_size, num_refine)
            obj = preds * sign[:, None]
            best_local = np.argmin(obj, axis=1)
            best_angles = local_angles[np.arange(batch_size), best_local]
            half_range = 2.0 * half_range / (num_refine - 1)

        best_dirs = np.stack([np.cos(best_angles), np.sin(best_angles)], axis=1)
        return best_dirs

    def _sample_best_gradients_3d(self, x_new, sdf, num_coarse=64,
                                  refine_steps=4, num_refine=16, initial_guess=None,
                                  chunk_size=200):
        batch_size = x_new.shape[0]
        sdf_flat = sdf.ravel()
        sign = np.where(sdf_flat > 0, 1.0, -1.0)

        def fibonacci_sphere(n):
            golden = (1 + np.sqrt(5)) / 2
            i = np.arange(n, dtype=float)
            theta = 2 * np.pi * i / golden
            phi = np.arccos(1 - 2 * (i + 0.5) / n)
            return np.stack([np.sin(phi)*np.cos(theta),
                             np.sin(phi)*np.sin(theta),
                             np.cos(phi)], axis=1)

        def tangent_frame(d):
            ref = np.where(np.abs(d[:, 0:1]) < 0.9,
                           np.broadcast_to([1., 0., 0.], d.shape).copy(),
                           np.broadcast_to([0., 1., 0.], d.shape).copy())
            t1 = np.cross(d, ref)
            t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
            t2 = np.cross(d, t1)
            return t1, t2

        def cone_dirs(best_dirs, half_angle, n):
            golden = (1 + np.sqrt(5)) / 2
            i = np.arange(n, dtype=float)
            r = half_angle * np.sqrt((i + 0.5) / n)
            alpha = 2 * np.pi * i / golden
            t1, t2 = tangent_frame(best_dirs)
            dirs = (np.cos(r)[None, :, None] * best_dirs[:, None, :]
                    + np.sin(r)[None, :, None] * (np.cos(alpha)[None, :, None] * t1[:, None, :]
                                                  + np.sin(alpha)[None, :, None] * t2[:, None, :]))
            return dirs

        if initial_guess is not None:
            valid_mask = ~np.isnan(initial_guess).any(axis=1)
            invalid_mask = ~valid_mask
        else:
            invalid_mask = np.ones(batch_size, dtype=bool)
            valid_mask = np.zeros(batch_size, dtype=bool)

        best_dirs = np.zeros((batch_size, 3))
        if np.any(valid_mask):
            best_dirs[valid_mask] = initial_guess[valid_mask]

        if np.any(invalid_mask):
            dirs = fibonacci_sphere(num_coarse)
            samples_invalid = x_new[invalid_mask, None, :] - sdf_flat[invalid_mask, None, None] * dirs[None, :, :]
            preds_invalid = self.predict(samples_invalid.reshape(-1, 3), chunk_size).reshape(np.sum(invalid_mask), num_coarse)
            obj_invalid = preds_invalid * sign[invalid_mask, None]
            best_idx_invalid = np.argmin(obj_invalid, axis=1)
            best_dirs[invalid_mask] = dirs[best_idx_invalid]

        half_angle = np.pi / np.sqrt(num_coarse)
        for _ in range(refine_steps):
            dirs_r = cone_dirs(best_dirs, half_angle, num_refine)
            samples = x_new[:, None, :] - sdf_flat[:, None, None] * dirs_r
            preds = self.predict(samples.reshape(-1, 3), chunk_size).reshape(batch_size, num_refine)
            obj = preds * sign[:, None]
            best_local = np.argmin(obj, axis=1)
            best_dirs = dirs_r[np.arange(batch_size), best_local]
            best_dirs /= np.linalg.norm(best_dirs, axis=1, keepdims=True)
            half_angle /= np.sqrt(num_refine)
        return best_dirs

class DuchonInterpolator(Interpolator):
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

    def fit(self, points, values, gradients=None, mask=None, force_recompute=False, hermite_interp=False, dist_threshold=0.2):
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
        dist_threshold (float, optional): If the number of points exceeds 5000, only keep points with absolute values less than this
            threshold to reduce time and memory usage. Default is 0.2.
        """
        if self.trained and not force_recompute:
            print(f"Interpolator is already trained. Use force_recompute=True to refit {points.shape[0]} points.")
            return
        # if DEBUG_TIME:
        #     start_time = time.time()
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
            close_mask = np.abs(self.values) < dist_threshold
            self.points = self.points[close_mask]
            self.values = self.values[close_mask]
            print(f"Warning: too many points, only keeping {len(self.values)} points with abs(value) < {dist_threshold} for fitting.")
        if not hermite_interp:
            self.alpha, self.p, self.q = self._compute_coefficients(self.points, self.values)
        self.trained = True
        # if DEBUG_TIME:
        #     end_time = time.time()
        #     print(f"  [Duchon fit] Fitting completed in {end_time - start_time:.3f} seconds. ({len(self.values)} points used)")
    
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

    def predict_gradients(self, x_new):
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
            pred_grad = self.predict_gradients(sample)[0]
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
            pred_grads = self.predict_gradients(samples)
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
    
    # Deprecated
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

    def predict_gradients(self, x_new, use_gradient_field=False):
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

class PUInterpolator(Interpolator):
    """
    Partition of Unity interpolator that accelerates RBF interpolation
    by decomposing the domain into overlapping patches with local solves.
    """
    def __init__(self, kernel='thin_plate', overlap=0.25,
                 min_points=10, max_points=200, partition='box'):
        """
        partition: 'fps' (farthest point sampling) or 'sphere' and 'box' (recursive median split).
        """
        self.kernel = kernel
        self.overlap = overlap
        self.min_points = min_points
        self.max_points = max_points
        self.partition = partition
        self.patches = []  # list of (center, radius, local_interpolator)
        self.trained = False
        # self.max_ext_points = max_points * (1 + max(overlap, 0.5))**3
        self.max_ext_points = 675
        if DEBUG_COVER:
            self.patch_cover_stats = [] # to record the covered patches for each prediction point for later analysis
        
    def _kdtree_partition(self, points, tree, is_box=False):
        """
        KD-tree style recursive median split, then extend each leaf into a ball.
        Same logic as the original _subdivide but returns the same
        (center, R_ext, ext_idx) format as _fps_partition.
        """
        def subdivide(indices):
            if len(indices) <= self.max_points:
                return [indices]
            pts = points[indices]
            spreads = pts.max(axis=0) - pts.min(axis=0)
            axis = int(np.argmax(spreads))
            median = np.median(pts[:, axis])
            left_mask = pts[:, axis] <= median
            if left_mask.all() or (~left_mask).all():
                return [indices]
            return subdivide(indices[left_mask]) + subdivide(indices[~left_mask])

        leaves = subdivide(np.arange(len(points)))
        patches_meta = []

        queue = deque(leaves)
        print(f"max_points={self.max_points}, max_ext_points={self.max_ext_points}")
        while queue:
            leaf_idx = queue.popleft()
            
            if is_box:
                leaf_pts = points[leaf_idx]
                spreads = leaf_pts.max(axis=0) - leaf_pts.min(axis=0)
                center = spreads * 0.5 + leaf_pts.min(axis=0)
                half_core = spreads * 0.5
                delta     = half_core.max() * self.overlap
                half_ext  = half_core + delta
                if self.overlap == 0.0:
                    ext_idx = leaf_idx
                else:
                    bsphere_r  = float(np.linalg.norm(half_ext))
                    candidates = np.array(tree.query_ball_point(center, bsphere_r), dtype=int)
                    in_box     = np.all(np.abs(points[candidates] - center) <= half_ext, axis=1)
                    ext_idx    = candidates[in_box]
                if len(leaf_idx) <= 2:
                    print(f"Warning: single-point leaf encountered at center {center}")
                # 超限则继续劈
                if len(ext_idx) > self.max_ext_points and len(leaf_idx) > 2:
                    pts = points[leaf_idx]
                    spreads = pts.max(axis=0) - pts.min(axis=0)
                    axis = int(np.argmax(spreads))
                    median = np.median(pts[:, axis])
                    left_mask = pts[:, axis] <= median
                    # TODO: fix the case where all points are on one side of the median (should be rare if max_points is reasonably large and data is not pathological)
                    if not left_mask.all() and not (~left_mask).all():
                        # queue.append(leaf_idx[left_mask])
                        # queue.append(leaf_idx[~left_mask])
                        queue.append(leaf_idx[left_mask])
                        queue.append(leaf_idx[~left_mask])  # ← 用 ~left_mask
                        continue  # 不生成 patch，改为处理子叶
                    else:
                        sort_pts = np.sort(pts, axis=0)
                        print(sort_pts[:, axis])
                        # export debug glb
                        import trimesh
                        from trimesh.visual.material import PBRMaterial
                        dbg = trimesh.Scene()
                        dbg.add_geometry(trimesh.PointCloud(leaf_pts[pts[:, axis]==median], colors=[0,255,0,255]), node_name='median_pts')
                        # leaf points (red)
                        dbg.add_geometry(trimesh.PointCloud(leaf_pts, colors=[255,0,0,255]), node_name='leaf_pts')
                        # ext points (green)
                        # ext_pts = points[ext_idx & ~leaf_idx]
                        ext_only = points[np.setdiff1d(ext_idx, leaf_idx)]
                        dbg.add_geometry(trimesh.PointCloud(ext_only, colors=[0,255,0,255]), node_name='ext_pts')
                        # core box (blue, semi-transparent)
                        s_core = trimesh.creation.box(extents=2 * half_core)
                        s_core.apply_translation(center)
                        s_core.visual.material = PBRMaterial(baseColorFactor=[0,0,255,80], alphaMode='BLEND')
                        dbg.add_geometry(s_core, node_name='core_box')
                        # extended box (yellow, semi-transparent)
                        s_ext = trimesh.creation.box(extents=2 * half_ext)
                        s_ext.apply_translation(center)
                        s_ext.visual.material = PBRMaterial(baseColorFactor=[255,255,0,60], alphaMode='BLEND')
                        dbg.add_geometry(s_ext, node_name='ext_box')
                        dbg.export('debug_leaf.glb')
                        print(f"  [debug] exported debug_leaf.glb  leaf={len(leaf_idx)} ext={len(ext_idx)}")

                    print(f"Warning: single-point leaf encountered at center {center} with core half_extent {np.linalg.norm(half_core):.4f}")
                    print(f"  Extending to {len(ext_idx)} points within half_extent {np.linalg.norm(half_ext):.4f} ({len(ext_idx) - 1} extra)")
                
                patches_meta.append((center, half_ext, ext_idx))

            else:
                leaf_pts = points[leaf_idx]
                center = leaf_pts.mean(axis=0)
                r_core = float(np.max(np.linalg.norm(leaf_pts - center, axis=1)))
                R_ext = r_core * (1.0 + self.overlap)
                ext_idx = np.array(tree.query_ball_point(center, R_ext), dtype=int)
                if len(leaf_idx) <= 2:
                    print(f"Warning: single-point leaf encountered at center {center} with core radius {r_core:.4f}")
                    print(f"  Extending to {len(ext_idx)} points within radius {R_ext:.4f} ({len(ext_idx) - 1} extra)")
                # 超限则继续劈
                if len(ext_idx) > self.max_ext_points and len(leaf_idx) > 2:
                    pts = points[leaf_idx]
                    spreads = pts.max(axis=0) - pts.min(axis=0)
                    axis = int(np.argmax(spreads))
                    median = np.median(pts[:, axis])
                    left_mask = pts[:, axis] <= median
                    if left_mask.all() or (~left_mask).all():
                        left_mask = pts[:, axis] < median  # 如果所有点都在一侧，说明数据分布极端，改为严格小于 median 作为左侧，避免出现空子叶
                    if left_mask.all() or (~left_mask).all():
                        left_mask = pts[:, axis] < center # 如果仍然所有点都在一侧，说明数据分布极端，改为小于 mean 作为左侧
                    if not left_mask.all() and not (~left_mask).all():
                        queue.append(leaf_idx[left_mask])
                        queue.append(leaf_idx[~left_mask])
                        continue
                    else:
                        sort_pts = np.sort(pts, axis=0)
                        print(sort_pts[:, axis])
                        # export debug glb
                        import trimesh
                        from trimesh.visual.material import PBRMaterial
                        dbg = trimesh.Scene()
                        # leaf points (red)
                        dbg.add_geometry(trimesh.PointCloud(leaf_pts, colors=[255,0,0,255]), node_name='leaf_pts')
                        # ext points (green)
                        # ext_pts = points[ext_idx & ~leaf_idx]
                        ext_only = points[np.setdiff1d(ext_idx, leaf_idx)]
                        dbg.add_geometry(trimesh.PointCloud(ext_only, colors=[0,255,0,255]), node_name='ext_pts')
                        # core box (blue, semi-transparent)
                        s_core = trimesh.creation.box(extents=[2*r_core, 2*r_core, 2*r_core])
                        s_core.apply_translation(center)
                        s_core.visual.material = PBRMaterial(baseColorFactor=[0,0,255,80], alphaMode='BLEND')
                        dbg.add_geometry(s_core, node_name='core_box')
                        # extended box (yellow, semi-transparent)
                        s_ext = trimesh.creation.box(extents=[2*R_ext, 2*R_ext, 2*R_ext])
                        s_ext.apply_translation(center)
                        s_ext.visual.material = PBRMaterial(baseColorFactor=[255,255,0,60], alphaMode='BLEND')
                        dbg.add_geometry(s_ext, node_name='ext_box')
                        dbg.export('debug_leaf.glb')
                        print(f"  [debug] exported debug_leaf.glb  leaf={len(leaf_idx)} ext={len(ext_idx)}")
                        # # print the spread and median info for debugging
                        # print(f"Warning: unable to split leaf at center {center} with core radius {r_core:.4f}")
                        # print(len(leaf_idx), "points in leaf, but cannot be split further.")
                        # print(f"  Spreads: {spreads}, median: {median}")
                        # print(f"  Left mask sum: {left_mask.sum()}, right mask sum: {right_mask.sum()}")
                        # print(f"  Extending to {len(ext_idx)} points within radius {R_ext:.4f} ({len(ext_idx) - 1} extra)")
                        # print(f"  Leaf points max min mean:\n{leaf_pts.max(axis=0)}\n{leaf_pts.min(axis=0)}\n{leaf_pts.mean(axis=0)}")

                patches_meta.append((center, R_ext, ext_idx))
        if not is_box:
            # 1. 提取所有中心点和半径转为 Numpy 数组，以便向量化加速
            centers = np.array([p[0] for p in patches_meta])
            radii = np.array([p[1] for p in patches_meta])
            
            # 2. 按照半径从大到小进行排序的索引
            sort_idx = np.argsort(-radii)
            
            # 3. 维护一个布尔掩码，记录哪些球被保留
            keep = np.ones(len(patches_meta), dtype=bool)
            
            for idx in range(len(sort_idx)):
                i = sort_idx[idx]
                
                # 如果这个球已经被更大的球吞并了，直接跳过
                if not keep[i]:
                    continue
                
                # 提取排在它后面的、且目前存活的更小的球的索引
                candidates = sort_idx[idx + 1:]
                candidates = candidates[keep[candidates]]
                
                if len(candidates) == 0:
                    break
                    
                # 4. 向量化计算当前大球到所有候选小球的距离
                dist = np.linalg.norm(centers[candidates] - centers[i], axis=1)
                
                # 5. 判断包含关系：中心距离 <= 大球半径 - 小球半径
                # 等价于 dist + r_small <= r_big
                contained = dist <= (radii[i] - radii[candidates])
                
                # 将所有被包含的小球标记为 False (剔除)
                keep[candidates[contained]] = False
                
            # 6. 生成最终清理后的 patches 列表
            patches_meta = [patches_meta[i] for i in range(len(patches_meta)) if keep[i]]
        return patches_meta

    @staticmethod
    def _wendland_weight(r, radius):
        """Wendland C2 compactly supported weight: (1 - r/R)^4 * (4r/R + 1), 0 outside."""
        s = np.clip(r / radius, 0.0, 1.0)
        return (1.0 - s) ** 4 * (4.0 * s + 1.0)

    @staticmethod
    def _box_weight(pts, center, half_ext):
        """
        Tensor-product Wendland C2 weight over an AABB.

        For each axis k: s_k = |x_k - c_k| / h_k
          phi_k = (1 - s_k)^4 * (4*s_k + 1)
        w(x) = prod_k phi_k,  zero outside the box.
        The product of C2 functions is still C2, and the support is exactly the box.
        """
        s = np.abs(pts - center) / half_ext        # (N, d)
        s = np.clip(s, 0.0, 1.0)
        phi = (1.0 - s) ** 4 * (4.0 * s + 1.0)    # (N, d)
        return phi.prod(axis=1)                     # (N,)

    def _fps_partition(self, points, tree):
        """
        Farthest Point Sampling partition: pick patch centers via FPS so they are
        spread evenly, then assign each center its k=max_points nearest neighbors as
        the core and extend by the overlap factor.
        n_patches is chosen so that each patch covers ~half of max_points on average,
        ensuring full coverage without gaps.
        """
        n = len(points)
        k = min(self.max_points, n)
        n_patches = max(1, int(np.ceil(n / (k / 2))))

        # --- FPS to pick n_patches center indices ---
        centroid = points.mean(axis=0)
        first = int(np.argmin(np.linalg.norm(points - centroid, axis=1)))
        centers_idx = [first]
        min_dists = np.linalg.norm(points - points[first], axis=1)

        for _ in range(n_patches - 1):
            next_idx = int(np.argmax(min_dists))
            centers_idx.append(next_idx)
            d = np.linalg.norm(points - points[next_idx], axis=1)
            min_dists = np.minimum(min_dists, d)

        # --- build patches ---
        patches_meta = []
        for ci in centers_idx:
            dists, core_idx = tree.query(points[ci], k=k)
            if np.isscalar(dists):
                dists = np.array([dists])
                core_idx = np.array([core_idx])
            r_core = float(dists[-1])
            center = points[core_idx].mean(axis=0)
            R_ext = r_core * (1.0 + self.overlap)
            ext_idx = np.array(tree.query_ball_point(center, R_ext), dtype=int)
            patches_meta.append((center, R_ext, ext_idx))

        return patches_meta
    
    def _deduplicate_sdf_points(self, points, values, tol=1e-8):
        # Deduplicate points that are very close to each other (within a small epsilon)
        # to avoid numerical issues in interpolation. We can use a KDTree for this.
        tree = KDTree(points)
        all_neighbors = tree.query_ball_tree(tree, r=tol)
        unique_mask = np.ones(len(points), dtype=bool)
        for i, neighbors in enumerate(all_neighbors):
            if not unique_mask[i]:
                continue
            for j in neighbors:
                if j > i:
                    unique_mask[j] = False
        return points[unique_mask], values[unique_mask]

    def fit(self, points, values, gradients=None, mask=None, dist_threshold=0.2):
        """
        Adaptively partition the domain and fit local interpolators on each patch.
        """
        if DEBUG_TIME: t0 = time.perf_counter()

        if gradients is not None:
            if mask is None:
                projections = points - values[:, np.newaxis] * gradients
            else:
                projections = points[mask] - values[mask, np.newaxis] * gradients[mask]
            points = np.vstack([points, projections])
            values = np.concatenate([values, np.zeros(len(projections))])
        if len(values) > 5000:
            close_mask = np.abs(values) < dist_threshold
            points = points[close_mask]
            values = values[close_mask]
            print(f"Warning: too many points, only keeping {len(values)} points with abs(value) < {dist_threshold} for fitting.")
        points, values = self._deduplicate_sdf_points(points, values)
        _, d = points.shape
        min_pts = max(self.min_points, d + 2)

        tree = KDTree(points)
        if DEBUG_TIME:
            t1 = time.perf_counter()
            print(f"  [PU fit] build KDTree: {t1-t0:.3f}s")
        
        if self.partition == 'sphere':
            patches_meta = self._kdtree_partition(points, tree)
        elif self.partition == 'box':
            patches_meta = self._kdtree_partition(points, tree, is_box=True)
            # patches_meta = self._grid_partition(points, tree)
        else:
            patches_meta = self._fps_partition(points, tree)
        if DEBUG_TIME:
            t2 = time.perf_counter()
            print(f"  [PU fit] greedy cover: {t2-t1:.3f}s  ({len(patches_meta)} patches)")

        self.patches = []
        if DEBUG_TIME: t_local_fit = 0; patch_sizes = []

        for center, R_ext, ext_idx in patches_meta:
            if len(ext_idx) < min_pts:
                continue

            local_pts = points[ext_idx]
            local_vals = values[ext_idx]
            if DEBUG_TIME: patch_sizes.append(len(ext_idx))

            if DEBUG_TIME: tf0 = time.perf_counter()
            interp = DuchonInterpolator(kernel=self.kernel)
            interp.fit(local_pts, local_vals)
            if DEBUG_TIME: t_local_fit += time.perf_counter() - tf0

            self.patches.append((center, R_ext, interp))

        if DEBUG_TIME:
            t3 = time.perf_counter()
            print(f"  [PU fit] patch loop: {t3-t2:.3f}s  (local fit: {t_local_fit:.3f}s)")
            if patch_sizes:
                print(f"  [PU fit] patch sizes: min={min(patch_sizes)}, max={max(patch_sizes)}, mean={np.mean(patch_sizes):.0f}")

        self._all_points = points
        self._all_values = values

        # Build KDTree on patch centers for fast query-to-patch lookup
        self._patch_type = 'box' if self.partition == 'box' else 'sphere'
        self._patch_centers = np.array([c for c, _, _ in self.patches])
        if self._patch_type == 'box':
            # Store bounding-sphere radius of each extended box for fallback queries
            self._patch_radii = np.array([float(np.linalg.norm(h)) for _, h, _ in self.patches])
        else:
            self._patch_radii = np.array([r for _, r, _ in self.patches])
        self._max_radius = self._patch_radii.max()
        self._patch_tree = KDTree(self._patch_centers)

        self.trained = True
        if DEBUG_TIME:
            print(f"  [PU fit] total: {time.perf_counter()-t0:.3f}s  ({len(self.patches)} patches)")
            self._visualize_patches(points, d)

    def _visualize_patches(self, points, d):
        """Visualize patches: 2D uses matplotlib, 3D exports a .glb file."""
        if d == 2:
            self._visualize_patches_2d(points)
        else:
            self._visualize_patches_3d(points)

    def _visualize_patches_2d(self, points):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        cmap = plt.cm.tab20
        for i, (center, radius, interp) in enumerate(self.patches):
            color = cmap(i % 20)
            circle = plt.Circle(center, radius, fill=True, alpha=0.15,
                                facecolor=color, edgecolor=color, linewidth=1.5)
            ax.add_patch(circle)
            # Plot points inside this patch
            dist = np.linalg.norm(points - center, axis=1)
            mask = dist < radius
            ax.scatter(points[mask, 0], points[mask, 1], s=2, color=color, alpha=0.5)
            ax.plot(*center, 'x', color=color, markersize=8)
        ax.set_aspect('equal')
        ax.autoscale()
        ax.set_title(f'PU Patches ({len(self.patches)} patches)')
        plt.savefig('pu_patches.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  [PU] saved patch visualization to pu_patches.png")

    def _visualize_patches_3d(self, points):
        try:
            import trimesh
            from trimesh.visual.material import PBRMaterial
        except ImportError:
            print("  [PU] trimesh not installed, skipping 3D visualization")
            return
        scene = trimesh.Scene()
        cmap_colors = [
            [31,119,180], [255,127,14], [44,160,44], [214,39,40],
            [148,103,189], [140,86,75], [227,119,194], [127,127,127],
            [188,189,34], [23,190,207], [174,199,232], [255,187,120],
            [152,223,138], [255,152,150], [197,176,213], [196,156,148],
            [247,182,210], [199,199,199], [219,219,141], [158,218,229],
        ]
        # Add points as small spheres
        sphere_meshes = []
        for pt in points:
            s = trimesh.primitives.Sphere(radius=0.003, center=pt, subdivisions=1).to_mesh()
            s.visual.vertex_colors = np.tile([255, 0, 0, 255], (len(s.vertices), 1))
            sphere_meshes.append(s)
        merged = trimesh.util.concatenate(sphere_meshes)
        scene.add_geometry(merged, node_name='points')
        # pc = trimesh.PointCloud(points, colors=[255, 0, 0, 255])
        # scene.add_geometry(pc, node_name=f'points')
        use_box = getattr(self, '_patch_type', 'sphere') == 'box'
        for i, (center, radius_or_half, _) in enumerate(self.patches):
            color = cmap_colors[i % len(cmap_colors)]
            if use_box:
                mesh = trimesh.creation.box(extents=2 * radius_or_half)
            else:
                mesh = trimesh.creation.icosphere(subdivisions=2, radius=radius_or_half)
            mesh.apply_translation(center)
            mesh.visual.material = PBRMaterial(baseColorFactor=[*color, 100], alphaMode='BLEND')
            scene.add_geometry(mesh, node_name=f'patch_{i}')
        scene.export(f'pu_patches_{len(self.patches)}.glb')
        print(f"  [PU] saved patch visualization to pu_patches_{len(self.patches)}.glb ({len(self.patches)} patches)")

    def predict(self, x_new, chunk_size=5000):
        """
        Predict values by blending local interpolators with Wendland weights.
        """
        if len(x_new) > chunk_size:
            return np.concatenate([self.predict(x_new[s:s + chunk_size], chunk_size)
                                   for s in range(0, len(x_new), chunk_size)])

        n = len(x_new)
        result = np.zeros(n)
        weight_sum = np.zeros(n)
        if DEBUG_COVER:
            self.patch_cover_stats = np.array([0] * n)
        use_box = getattr(self, '_patch_type', 'sphere') == 'box'
        for center, radius_or_half, interp in self.patches:
            if use_box:
                idx = np.where(np.all(np.abs(x_new - center) <= radius_or_half, axis=1))[0]
                if len(idx) == 0:
                    continue
                pts = x_new[idx]
                w   = self._box_weight(pts, center, radius_or_half)
            else:
                dist = np.linalg.norm(x_new - center, axis=1)
                idx  = np.where(dist <= radius_or_half)[0]
                if len(idx) == 0:
                    continue
                w = self._wendland_weight(dist[idx], radius_or_half)
                pts = x_new[idx]

            v = interp.predict(pts)
            result[idx] += w * v
            weight_sum[idx] += w
            if DEBUG_COVER:
                self.patch_cover_stats[idx] += 1
            

        safe = weight_sum > 0
        result[safe] /= weight_sum[safe]

        # Fallback: for uncovered points, use nearest patch by surface distance
        uncovered = ~safe
        if np.any(uncovered):
            uncov_pts = x_new[uncovered]
            _, nearest = self._patch_tree.query(uncov_pts)
            for pi in np.unique(nearest):
                pts_mask = nearest == pi
                _, _, interp = self.patches[pi]
                result[np.where(uncovered)[0][pts_mask]] = interp.predict(uncov_pts[pts_mask])

        if DEBUG_COVER:
            if np.sum(uncovered) > 200 and len(x_new) < chunk_size:
                filtered_cover_stats = self.patch_cover_stats[safe]
                if len(filtered_cover_stats) == 0:
                    print(f"  [PU predict] all {n} points uncovered")
                    return result
                print(f"  [PU predict] patch cover stats: "
                    f"min={min(filtered_cover_stats)}, "
                    f"max={max(filtered_cover_stats)}, "
                    f"mean={np.mean(filtered_cover_stats):.2f}"
                    f"  (uncovered: {np.sum(uncovered)}/{n} points)")
                # export uncovered points for debugging
                import trimesh
                from trimesh.visual.material import PBRMaterial
                dbg = trimesh.Scene()
                # dbg.add_geometry(trimesh.PointCloud(x_new[safe], colors=[0,255,0,255]), node_name='covered')
                if False:
                    sphere_meshes = []
                    for pt in x_new[self.patch_cover_stats <= 1]:
                        s = trimesh.primitives.Sphere(radius=0.003, center=pt, subdivisions=1).to_mesh()
                        s.visual.vertex_colors = np.tile([255, 0, 0, 255], (len(s.vertices), 1))
                        sphere_meshes.append(s)
                    merged = trimesh.util.concatenate(sphere_meshes)
                    dbg.add_geometry(merged, node_name='uncovered')
                else:
                    dbg.add_geometry(trimesh.PointCloud(x_new[uncovered], colors=[255,0,0,255]), node_name='uncovered')
                dbg.export('pu_uncovered.glb')
                print(f"  [PU] exported uncovered points to pu_uncovered.glb")  

        return result
    
    def predict_gradients(self, x_new, chunk_size=5000):
        """
        Predict gradients by blending local interpolators with Wendland weights.
        """
        if len(x_new) > chunk_size:
            return np.concatenate([self.predict_gradients(x_new[s:s + chunk_size], chunk_size)
                                   for s in range(0, len(x_new), chunk_size)])

        n = len(x_new)
        d = x_new.shape[1]
        result = np.zeros((n, d))
        weight_sum = np.zeros(n)

        use_box = getattr(self, '_patch_type', 'sphere') == 'box'            
        for center, radius_or_half, interp in self.patches:
            if use_box:
                idx = np.where(np.all(np.abs(x_new - center) <= radius_or_half, axis=1))[0]
                if len(idx) == 0:
                    continue
                pts = x_new[idx]
                w   = self._box_weight(pts, center, radius_or_half)
            else:
                dist = np.linalg.norm(x_new - center, axis=1)
                idx  = np.where(dist <= radius_or_half)[0]
                if len(idx) == 0:
                    continue
                w   = self._wendland_weight(dist[idx], radius_or_half)
                pts = x_new[idx]

            g = interp.predict_gradients(pts)
            result[idx] += w[:, np.newaxis] * g
            weight_sum[idx] += w

        safe = weight_sum > 0
        result[safe] /= weight_sum[safe, np.newaxis]

        # Fallback: for uncovered points, use nearest patch to predict
        uncovered = ~safe
        if np.any(uncovered):
            uncov_pts = x_new[uncovered]
            _, nearest = self._patch_tree.query(uncov_pts)
            for patch_idx in np.unique(nearest):
                pts_mask = nearest == patch_idx
                _, _, interp = self.patches[patch_idx]
                result[np.where(uncovered)[0][pts_mask]] = interp.predict_gradients(uncov_pts[pts_mask])

        return result


