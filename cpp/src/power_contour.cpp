#include "power_contour.h"

#include <vector>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#ifdef HAVE_MES_CONTACT
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Robust_weighted_circumcenter_filtered_traits_3.h>
#include <CGAL/Regular_triangulation_vertex_base_3.h>
#include <CGAL/Regular_triangulation_cell_base_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/Triangulation_cell_base_with_info_3.h>
#include <CGAL/Triangulation_data_structure_3.h>
#include <CGAL/Regular_triangulation_3.h>
#endif

namespace power_contour {

void extract(
    const Eigen::MatrixXd& points,
    const Eigen::VectorXd& sdf_values,
    const Eigen::MatrixXd& tangent_pts,
    bool clip_to_bbox,
    bool verbose,
    Eigen::MatrixXd& out_V,
    Eigen::MatrixXi& out_F,
    Eigen::MatrixXi* out_sites)
{
    const int N = (int)points.rows();
    if (sdf_values.size() != N || tangent_pts.rows() != N)
        throw std::invalid_argument("power_contour::extract: points, sdf_values and "
                                    "tangent_pts must have the same number of rows");
    out_V.resize(0, 3);
    out_F.resize(0, 3);
    if (out_sites) out_sites->resize(0, 2);
    if (N == 0) return;

#ifdef HAVE_MES_CONTACT
    // Grid samples are highly degenerate, so many cells are nearly flat and a plain
    // double weighted circumcenter (the power center) lands far off. This traits
    // class can construct it exactly (force_exact below); its own switch to exact
    // only triggers on a zero denominator, which is not enough here.
    using K   = CGAL::Robust_weighted_circumcenter_filtered_traits_3<
                    CGAL::Exact_predicates_inexact_constructions_kernel>;
    // vertex info = signed site id: +(1+k) / -(1+k) for a positive / negative site k,
    // k = i for sample i and N + i for the tangent point of sample i
    using Vb  = CGAL::Triangulation_vertex_base_with_info_3<int, K,
                    CGAL::Regular_triangulation_vertex_base_3<K>>;
    // cell info = index of the cell's power center in the output vertex array (-1 = unset)
    using Cb  = CGAL::Triangulation_cell_base_with_info_3<int, K,
                    CGAL::Regular_triangulation_cell_base_3<K>>;
    using Tds = CGAL::Triangulation_data_structure_3<Vb, Cb>;
    using Rt  = CGAL::Regular_triangulation_3<K, Tds>;
    using WP  = Rt::Weighted_point;
    using BP  = Rt::Bare_point;

    // Sites: N samples (weight s^2, sign of s) + valid tangent points (weight 0,
    // opposite sign of their sample).
    std::vector<std::pair<WP, int>> wpts;
    wpts.reserve(2 * N);
    int n_tp = 0;
    for (int i = 0; i < N; i++) {
        double s = sdf_values(i);
        int sgn = s >= 0 ? 1 : -1;
        wpts.emplace_back(WP(BP(points(i, 0), points(i, 1), points(i, 2)), s * s),
                          sgn * (1 + i));
        if (!tangent_pts.row(i).allFinite()) continue;
        wpts.emplace_back(WP(BP(tangent_pts(i, 0), tangent_pts(i, 1), tangent_pts(i, 2)), 0.0),
                          -sgn * (1 + N + i));
        n_tp++;
    }
    Rt rt(wpts.begin(), wpts.end());

    Eigen::Vector3d bmin = points.colwise().minCoeff().transpose();
    Eigen::Vector3d bmax = points.colwise().maxCoeff().transpose();
    auto inside_bbox = [&](const Eigen::Vector3d& p) {
        return (p.array() >= bmin.array()).all() && (p.array() <= bmax.array()).all();
    };

    for (auto cit = rt.all_cells_begin(); cit != rt.all_cells_end(); ++cit) cit->info() = -1;
    std::vector<Eigen::Vector3d> V;
    // power center (weighted circumcenter) of a finite cell, computed once, exactly
    auto wcc = rt.geom_traits().construct_weighted_circumcenter_3_object();
    auto center_index = [&](Rt::Cell_handle c) {
        if (c->info() < 0) {
            BP p = wcc(c->vertex(0)->point(), c->vertex(1)->point(),
                       c->vertex(2)->point(), c->vertex(3)->point(), true);
            c->info() = (int)V.size();
            V.emplace_back(p.x(), p.y(), p.z());
        }
        return c->info();
    };

    std::vector<Eigen::Vector3i> F;
    std::vector<Eigen::Vector2i> sites;  // per triangle: (positive site, negative site)
    std::vector<int> ring;
    int n_cross = 0, n_unbounded = 0, n_clipped = 0;
    for (auto eit = rt.finite_edges_begin(); eit != rt.finite_edges_end(); ++eit) {
        auto v1 = eit->first->vertex(eit->second);
        auto v2 = eit->first->vertex(eit->third);
        if ((v1->info() > 0) == (v2->info() > 0)) continue;
        n_cross++;

        // ring of cells around the edge = the dual face's polygon
        ring.clear();
        bool bounded = true;
        auto cc = rt.incident_cells(*eit), done = cc;
        do {
            if (rt.is_infinite(cc)) { bounded = false; break; }
            ring.push_back(center_index(cc));
        } while (++cc != done);
        if (!bounded) { n_unbounded++; continue; }
        if (clip_to_bbox) {
            bool ok = true;
            for (int k : ring) ok = ok && inside_bbox(V[k]);
            if (!ok) { n_clipped++; continue; }
        }

        // orient so the normal points from the negative site to the positive one
        Eigen::Vector3d pos(v1->point().x(), v1->point().y(), v1->point().z());
        Eigen::Vector3d neg(v2->point().x(), v2->point().y(), v2->point().z());
        if (v1->info() < 0) std::swap(pos, neg);
        Eigen::Vector3d nrm = Eigen::Vector3d::Zero();
        for (size_t k = 1; k + 1 < ring.size(); k++)
            nrm += (V[ring[k]] - V[ring[0]]).cross(V[ring[k + 1]] - V[ring[0]]);
        bool flip = nrm.dot(pos - neg) < 0;
        int sp = std::abs(v1->info()) - 1, sn = std::abs(v2->info()) - 1;
        if (v1->info() < 0) std::swap(sp, sn);
        for (size_t k = 1; k + 1 < ring.size(); k++) {
            if (flip) F.emplace_back(ring[0], ring[k + 1], ring[k]);
            else      F.emplace_back(ring[0], ring[k], ring[k + 1]);
            sites.emplace_back(sp, sn);
        }
    }

    // keep only the power centers the faces use
    std::vector<int> remap(V.size(), -1);
    int m = 0;
    for (auto& f : F)
        for (int k = 0; k < 3; k++) {
            if (remap[f[k]] < 0) remap[f[k]] = m++;
            f[k] = remap[f[k]];
        }
    out_V.resize(m, 3);
    for (size_t i = 0; i < V.size(); i++)
        if (remap[i] >= 0) out_V.row(remap[i]) = V[i].transpose();
    out_F.resize((int)F.size(), 3);
    for (size_t i = 0; i < F.size(); i++) out_F.row(i) = F[i].transpose();
    if (out_sites) {
        out_sites->resize((int)sites.size(), 2);
        for (size_t i = 0; i < sites.size(); i++) out_sites->row(i) = sites[i].transpose();
    }

    if (verbose)
        std::cout << "[power_contour] sites: " << N << " samples + " << n_tp
                  << " tangent pts (" << rt.number_of_vertices() << " non-hidden); "
                  << n_cross << " sign-crossing edges, " << n_unbounded << " unbounded, "
                  << n_clipped << " clipped; " << m << " verts, " << F.size() << " tris\n";
#else
    (void)clip_to_bbox; (void)verbose; (void)out_sites;
    throw std::runtime_error("power_contour::extract needs CGAL (HAVE_MES_CONTACT); rebuild with CGAL.");
#endif
}

}  // namespace power_contour
