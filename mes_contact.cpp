/*
 * mes_contact.cpp
 *
 * Pybind11 wrapper for maximal empty sphere contact points.
 * Given SDF sample points and values, returns the contact point and
 * outward normal of the largest adjacent maximal empty sphere for each
 * input point. Points without a contact sphere get NaN rows.
 *
 * Compile (run from sdfgradients/):
 *   python3 build_mes_contact.py
 *
 * Usage in Python:
 *   import mes_contact
 *   pts, nrm = mes_contact.contact_points_from_sdf(points, sdf)
 *   # pts, nrm: (N, 3) float64, NaN where no contact sphere found
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Maximal_empty_spheres/maximal_empty_spheres.h>
#include <CGAL/Dimension.h>

#include <Eigen/Core>

#include <vector>
#include <cmath>
#include <limits>

namespace py = pybind11;

// Process one sign-group of spheres and fill in the (N,3) output arrays.
// G      : (M, 4) matrix of x,y,z,r for this group (r may be pos or neg)
// orig   : original index in [0, N) for each row of G
// out_pts: (N, 3) output, filled at rows given by orig
// out_nrm: (N, 3) output, filled at rows given by orig
// filter_bbox: skip contact spheres outside the axis-aligned bbox of G
static void process_group(
    const Eigen::MatrixXd& G,
    const std::vector<int>& orig,
    py::detail::unchecked_mutable_reference<double, 2>& out_pts,
    py::detail::unchecked_mutable_reference<double, 2>& out_nrm,
    bool filter_bbox,
    int  debug_level)
{
    int M = (int)G.rows();
    if (M == 0) return;

    // maximal_empty_spheres requires non-negative radii
    Eigen::MatrixXd G_abs = G;
    G_abs.col(3) = G.col(3).array().abs();

    Eigen::MatrixXd result;         // contact spheres (K, 4)
    Eigen::MatrixXi contact_indices; // (K, ncp_max): which input spheres each contacts
    CGAL::maximal_empty_spheres<CGAL::Dimension_tag<3>>(
        G_abs, result, &contact_indices, /*atol=*/1e-8, debug_level,
        /*ncp_max=*/M, /*cone_filter=*/true);

    if (debug_level > 0) {
        std::cout << "[process_group] M=" << M
                  << "  CGAL returned " << result.rows() << " contact spheres"
                  << std::endl;
    }

    // Bounding box of input sphere centers (for optional filtering)
    Eigen::RowVector3d bb_lo = G_abs.leftCols(3).colwise().minCoeff();
    Eigen::RowVector3d bb_hi = G_abs.leftCols(3).colwise().maxCoeff();

    // For each input sphere, find the adjacent contact sphere with the
    // largest absolute radius (mirrors the logic in contact_points_from_signed_distances.h)
    Eigen::VectorXi cp_idx = Eigen::VectorXi::Constant(M, -1);
    Eigen::VectorXd cp_r   = Eigen::VectorXd::Constant(M, -1.0);

    int n_bbox_filtered = 0;
    for (int i = 0; i < result.rows(); i++) {
        if (filter_bbox) {
            // Skip contact spheres whose center lies outside the input bbox
            bool inside = true;
            for (int d = 0; d < 3; d++) {
                if (result(i, d) <= bb_lo(d) || result(i, d) > bb_hi(d)) {
                    inside = false; break;
                }
            }
            if (!inside) { n_bbox_filtered++; continue; }
        }
        double r = std::fabs(result(i, 3)); // contact spheres have negative radius
        for (int j = 0; j < contact_indices.cols(); j++) {
            int n = contact_indices(i, j);
            if (n < 0 || n >= M) continue;
            if (cp_idx(n) < 0 || cp_r(n) < r) {
                cp_idx(n) = i;
                cp_r(n)   = r;
            }
        }
    }

    if (debug_level > 0) {
        int n_assigned = 0;
        for (int i = 0; i < M; i++) if (cp_idx(i) >= 0) n_assigned++;
        std::cout << "[process_group] bbox_filtered=" << n_bbox_filtered
                  << "  assigned=" << n_assigned << "/" << M << std::endl;
    }

    // Compute contact point and normal for each sphere that has a match
    for (int i = 0; i < M; i++) {
        if (cp_idx(i) < 0) continue;
        int oi = orig[i];

        double cx = G(i, 0), cy = G(i, 1), cz = G(i, 2), r = G(i, 3);
        double rx = result(cp_idx(i), 0);
        double ry = result(cp_idx(i), 1);
        double rz = result(cp_idx(i), 2);

        // Unit vector from input sphere center toward contact sphere center
        double dx = rx - cx, dy = ry - cy, dz = rz - cz;
        double vl = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (vl < 1e-15) continue;
        dx /= vl; dy /= vl; dz /= vl;

        // Contact point: on surface of input sphere in direction of contact sphere
        double ar = std::fabs(r);
        out_pts(oi, 0) = cx + ar * dx;
        out_pts(oi, 1) = cy + ar * dy;
        out_pts(oi, 2) = cz + ar * dz;

        // Normal convention: for outside spheres (r >= 0) normal points inward (-D),
        // for inside spheres (r < 0) normal points outward (+D)
        double sign = (r >= 0) ? -1.0 : 1.0;
        out_nrm(oi, 0) = sign * dx;
        out_nrm(oi, 1) = sign * dy;
        out_nrm(oi, 2) = sign * dz;
    }
}


py::tuple contact_points_from_sdf(
    py::array_t<double, py::array::c_style | py::array::forcecast> points_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> sdf_arr,
    bool filter_bbox  = true,
    int  debug_level  = 0)
{
    auto pts = points_arr.unchecked<2>();
    auto sdf = sdf_arr.unchecked<1>();
    int N = (int)pts.shape(0);

    // Split into positive (outside) and negative (inside) groups,
    // tracking original indices so we can write back into (N, 3) arrays.
    std::vector<int> pos_orig, neg_orig;
    for (int i = 0; i < N; i++) {
        if (sdf(i) >= 0) pos_orig.push_back(i);
        else             neg_orig.push_back(i);
    }

    Eigen::MatrixXd Gp((int)pos_orig.size(), 4);
    Eigen::MatrixXd Gn((int)neg_orig.size(), 4);
    for (int k = 0; k < (int)pos_orig.size(); k++) {
        int i = pos_orig[k];
        Gp.row(k) = Eigen::RowVector4d(pts(i,0), pts(i,1), pts(i,2),  sdf(i));
    }
    for (int k = 0; k < (int)neg_orig.size(); k++) {
        int i = neg_orig[k];
        Gn.row(k) = Eigen::RowVector4d(pts(i,0), pts(i,1), pts(i,2),  sdf(i));
    }

    // Allocate output arrays, initialized to NaN
    const double nan = std::numeric_limits<double>::quiet_NaN();
    py::array_t<double> out_pts({N, 3});
    py::array_t<double> out_nrm({N, 3});
    auto p = out_pts.mutable_unchecked<2>();
    auto n = out_nrm.mutable_unchecked<2>();
    for (int i = 0; i < N; i++)
        for (int j = 0; j < 3; j++)
            p(i, j) = n(i, j) = nan;

    process_group(Gp, pos_orig, p, n, filter_bbox, debug_level);
    process_group(Gn, neg_orig, p, n, filter_bbox, debug_level);

    return py::make_tuple(out_pts, out_nrm);
}


PYBIND11_MODULE(mes_contact, m) {
    m.doc() = "Maximal empty sphere contact points from signed distances";

    m.def("contact_points_from_sdf", &contact_points_from_sdf,
          py::arg("points"),
          py::arg("sdf"),
          py::arg("filter_bbox")  = true,
          py::arg("debug_level")  = 0,
          R"(Compute contact points of maximal empty spheres.

Args:
    points      : (N, 3) float64 — sample point positions
    sdf         : (N,)   float64 — signed distance values
                  (positive = outside surface, negative = inside)
    filter_bbox : if True, ignore contact spheres outside the input bbox
    debug_level : verbosity (0 = quiet)

Returns:
    (contact_pts, normals) — each (N, 3) float64.
    Rows are NaN for points that have no adjacent contact sphere.
    Normal convention: points away from the surface (inward for outside
    spheres, outward for inside spheres).)");
}
