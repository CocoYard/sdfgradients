#include "mes_contact_core.h"

#ifdef HAVE_MES_CONTACT

#include <CGAL/Maximal_empty_spheres/maximal_empty_spheres.h>
#include <CGAL/Dimension.h>

#include <vector>
#include <cmath>
#include <limits>
#include <iostream>
#include <fstream>
#include <sys/resource.h>

namespace {
double rss_gb() {
    struct rusage u;
    getrusage(RUSAGE_SELF, &u);
    // ru_maxrss is in bytes on macOS
    return (double)u.ru_maxrss / (1024.0 * 1024.0 * 1024.0);
}
}


namespace mes_contact_core {

static void process_group(
    const Eigen::MatrixXd& G,
    const std::vector<int>& orig,
    Eigen::MatrixXd& out_pts,
    Eigen::MatrixXd& out_nrm,
    bool filter_bbox,
    int  debug_level)
{
    int M = (int)G.rows();
    if (M == 0) return;

    Eigen::MatrixXd G_abs = G;
    G_abs.col(3) = G.col(3).array().abs();

    Eigen::MatrixXd result;
    Eigen::MatrixXi contact_indices;

    std::cout << "[MES] process_group M=" << M
              << "  Eigen::nbThreads=" << Eigen::nbThreads()
              << "  RSS before CGAL=" << rss_gb() << " GB\n" << std::flush;

    CGAL::maximal_empty_spheres<CGAL::Dimension_tag<3>>(
        G_abs, result, &contact_indices, /*atol=*/1e-8, debug_level,
        /*ncp_max=*/10, /*cone_filter=*/true);

    std::cout << "[MES] process_group M=" << M
              << "  result.rows=" << result.rows()
              << "  RSS after  CGAL=" << rss_gb() << " GB\n" << std::flush;

    if (debug_level > 0) {
        std::cout << "[mes_contact_core] M=" << M
                  << "  CGAL returned " << result.rows() << " contact spheres\n";
    }

    Eigen::RowVector3d bb_lo = G_abs.leftCols(3).colwise().minCoeff();
    Eigen::RowVector3d bb_hi = G_abs.leftCols(3).colwise().maxCoeff();

    Eigen::VectorXi cp_idx = Eigen::VectorXi::Constant(M, -1);
    Eigen::VectorXd cp_r   = Eigen::VectorXd::Constant(M, -1.0);

    int n_bbox_filtered = 0;
    for (int i = 0; i < result.rows(); i++) {
        if (filter_bbox) {
            bool inside = true;
            for (int d = 0; d < 3; d++) {
                if (result(i, d) <= bb_lo(d) || result(i, d) > bb_hi(d)) {
                    inside = false; break;
                }
            }
            if (!inside) { n_bbox_filtered++; continue; }
        }
        double r = std::fabs(result(i, 3));
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
        std::cout << "[mes_contact_core] bbox_filtered=" << n_bbox_filtered
                  << "  assigned=" << n_assigned << "/" << M << "\n";
    }

    for (int i = 0; i < M; i++) {
        if (cp_idx(i) < 0) continue;
        int oi = orig[i];
        double cx = G(i, 0), cy = G(i, 1), cz = G(i, 2), r = G(i, 3);
        double rx = result(cp_idx(i), 0);
        double ry = result(cp_idx(i), 1);
        double rz = result(cp_idx(i), 2);

        double dx = rx - cx, dy = ry - cy, dz = rz - cz;
        double vl = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (vl < 1e-15) continue;
        dx /= vl; dy /= vl; dz /= vl;

        double ar = std::fabs(r);
        out_pts(oi, 0) = cx + ar * dx;
        out_pts(oi, 1) = cy + ar * dy;
        out_pts(oi, 2) = cz + ar * dz;

        double sign = (r >= 0) ? -1.0 : 1.0;
        out_nrm(oi, 0) = sign * dx;
        out_nrm(oi, 1) = sign * dy;
        out_nrm(oi, 2) = sign * dz;
    }
}

void contact_points_from_sdf(
    const Eigen::MatrixXd& points,
    const Eigen::VectorXd& sdf_values,
    bool filter_bbox,
    int  debug_level,
    Eigen::MatrixXd& out_pts,
    Eigen::MatrixXd& out_normals)
{
    // Eigen+OpenMP in this process spawns per-thread scratch buffers that
    // push CGAL MES memory usage into triple digits of GB. Force single-threaded
    // Eigen for the duration of this call; restore on exit.
    int saved_threads = Eigen::nbThreads();
    Eigen::setNbThreads(1);
    struct RestoreThreads {
        int v;
        ~RestoreThreads() { Eigen::setNbThreads(v); }
    } restore{saved_threads};

    int N = (int)points.rows();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    out_pts     = Eigen::MatrixXd::Constant(N, 3, nan);
    out_normals = Eigen::MatrixXd::Constant(N, 3, nan);
    if (N == 0) return;

    // Dump exact inputs so we can reproduce standalone with test_mes_bench.
    {
        std::ofstream f("/tmp/mes_input.csv");
        f.precision(17);
        for (int i = 0; i < N; i++) {
            f << points(i,0) << "," << points(i,1) << ","
              << points(i,2) << "," << sdf_values(i) << "\n";
        }
        std::cout << "[mes_contact_core] dumped " << N
                  << " rows to /tmp/mes_input.csv\n";
    }

    std::vector<int> pos_orig, neg_orig;
    for (int i = 0; i < N; i++) {
        if (sdf_values(i) >= 0) pos_orig.push_back(i);
        else                    neg_orig.push_back(i);
    }

    Eigen::MatrixXd Gp((int)pos_orig.size(), 4);
    Eigen::MatrixXd Gn((int)neg_orig.size(), 4);
    for (int k = 0; k < (int)pos_orig.size(); k++) {
        int i = pos_orig[k];
        Gp.row(k) = Eigen::RowVector4d(points(i,0), points(i,1), points(i,2), sdf_values(i));
    }
    for (int k = 0; k < (int)neg_orig.size(); k++) {
        int i = neg_orig[k];
        Gn.row(k) = Eigen::RowVector4d(points(i,0), points(i,1), points(i,2), sdf_values(i));
    }

    process_group(Gp, pos_orig, out_pts, out_normals, filter_bbox, debug_level);
    process_group(Gn, neg_orig, out_pts, out_normals, filter_bbox, debug_level);
}

}  // namespace mes_contact_core

#else  // !HAVE_MES_CONTACT

#include <limits>
#include <iostream>

namespace mes_contact_core {
void contact_points_from_sdf(
    const Eigen::MatrixXd& points,
    const Eigen::VectorXd& /*sdf_values*/,
    bool /*filter_bbox*/,
    int  /*debug_level*/,
    Eigen::MatrixXd& out_pts,
    Eigen::MatrixXd& out_normals)
{
    int N = (int)points.rows();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    out_pts     = Eigen::MatrixXd::Constant(N, 3, nan);
    out_normals = Eigen::MatrixXd::Constant(N, 3, nan);
    static bool warned = false;
    if (!warned) {
        std::cerr << "[mes_contact_core] HAVE_MES_CONTACT not defined; "
                     "returning NaN normals. Rebuild with CGAL to enable.\n";
        warned = true;
    }
}
}  // namespace mes_contact_core

#endif  // HAVE_MES_CONTACT
