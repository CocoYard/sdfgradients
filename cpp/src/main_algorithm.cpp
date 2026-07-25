#include "main_algorithm.h"
#include "duchon_interpolator.h"
#include "pu_interpolator.h"
#include "optimization.h"
#include "visibility.h"
#include "kdtree.h"
#include "thread_policy.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <limits>
#include <algorithm>
#include <chrono>

namespace sdf {
using clk = std::chrono::steady_clock;

static auto ms_since = [](const clk::time_point& t) {
    return std::chrono::duration<double, std::milli>(clk::now() - t).count();
};
// Forward declaration (defined after init_gradients_by_degenerate_pts)
static void export_short_arcs_ply(
    const Eigen::MatrixXd& sdf_points,
    const Options& options,
    const std::string& out_dir = "out",
    bool use_new = true
);

// ── filter_degenerate_pts ───────────────────────────────────────────

static void filter_degenerate_pts(
    std::unordered_map<int, std::vector<Eigen::Vector3d>>& degenerate_pts,
    const Interpolator& interpolator,
    bool verbose = true,
    double dist_tol = 0.1,
    double spatial_dedup_tol = 1e-4)
{
    constexpr double dedup_tol = 1e-8;
    std::vector<int> to_remove;

    // Step 1: per-sphere dedup. Collapse clusters within dedup_tol. For still-
    // ambiguous entries (>1 distinct cluster) we defer the "pick smallest |pred|"
    // decision to a single batched predict() call below.
    std::vector<int> ambig_idx;            // sphere ids with >1 distinct cluster
    std::vector<std::vector<Eigen::Vector3d>> ambig_candidates;

    for (auto& [idx, pts] : degenerate_pts) {
        if (pts.empty()) { to_remove.push_back(idx); continue; }
        if ((int)pts.size() == 1) continue;

        std::vector<Eigen::Vector3d> uniq;
        for (const auto& p : pts) {
            bool dup = false;
            for (const auto& u : uniq) {
                if ((p - u).norm() < dedup_tol) { dup = true; break; }
            }
            if (!dup) uniq.push_back(p);
        }
        if (uniq.size() == 1) {
            pts = std::move(uniq);
        } else {
            ambig_idx.push_back(idx);
            ambig_candidates.push_back(std::move(uniq));
        }
    }

    // Step 2: resolve ambiguous entries in one batched predict().
    if (!ambig_idx.empty()) {
        int total = 0;
        for (const auto& c : ambig_candidates) total += (int)c.size();
        Eigen::MatrixXd P(total, 3);
        int row = 0;
        for (const auto& c : ambig_candidates)
            for (const auto& p : c) P.row(row++) = p.transpose();
        Eigen::VectorXd preds = interpolator.predict(P);

        int off = 0;
        for (size_t i = 0; i < ambig_idx.size(); i++) {
            const auto& c = ambig_candidates[i];
            int best = 0;
            double best_abs = std::abs(preds(off));
            for (int k = 1; k < (int)c.size(); k++) {
                double a = std::abs(preds(off + k));
                if (a < best_abs) { best_abs = a; best = k; }
            }
            degenerate_pts[ambig_idx[i]] = { c[best] };
            off += (int)c.size();
        }
    }

    // Step 3: batched surface-proximity check on the (now single) representative.
    std::vector<int> keep_idx;
    keep_idx.reserve(degenerate_pts.size());
    for (const auto& [idx, pts] : degenerate_pts) {
        if (std::find(to_remove.begin(), to_remove.end(), idx) == to_remove.end())
            keep_idx.push_back(idx);
    }
    if (!keep_idx.empty()) {
        Eigen::MatrixXd Q((int)keep_idx.size(), 3);
        for (int i = 0; i < (int)keep_idx.size(); i++)
            Q.row(i) = degenerate_pts[keep_idx[i]][0].transpose();
        Eigen::VectorXd preds = interpolator.predict(Q);
        for (int i = 0; i < (int)keep_idx.size(); i++) {
            if (std::abs(preds(i)) > dist_tol) {
                // std::cout << "Degenerate point " << keep_idx[i]
                //           << " is too far from the surface with predicted sdf "
                //           << preds(i) << ", removing it.\n";
                to_remove.push_back(keep_idx[i]);
            }
        }
    }

    // Step 4: cross-sphere spatial dedup. Different spheres' degenerate
    // projections often land on the same arc/edge, creating huge
    // near-duplicate clusters that blow up the downstream PU patch sizes
    // (Duchon is O(m^3)). Keep one representative per spatial cluster;
    // flag the duplicates for removal — they'll just drop their init
    // gradient and be treated as non-degenerate downstream.
    std::vector<int> survivors;
    survivors.reserve(keep_idx.size());
    {
        std::unordered_set<int> rm(to_remove.begin(), to_remove.end());
        for (int idx : keep_idx)
            if (!rm.count(idx)) survivors.push_back(idx);
    }
    if ((int)survivors.size() > 1) {
        Eigen::MatrixXd R((int)survivors.size(), 3);
        for (int i = 0; i < (int)survivors.size(); i++)
            R.row(i) = degenerate_pts[survivors[i]][0].transpose();
        KDTree3D stree(R);
        std::vector<char> keep_flag(survivors.size(), 1);
        int dedup_cnt = 0;
        for (int i = 0; i < (int)survivors.size(); i++) {
            if (!keep_flag[i]) continue;
            Eigen::Vector3d pt = R.row(i);
            auto neigh = stree.query_ball_point(pt, spatial_dedup_tol);
            for (int j : neigh) {
                if (j > i && keep_flag[j]) {
                    keep_flag[j] = 0;
                    to_remove.push_back(survivors[j]);
                    dedup_cnt++;
                }
            }
        }
        if (dedup_cnt > 0 && verbose)
            std::cout << "Cross-sphere dedup removed " << dedup_cnt
                      << " near-duplicate degenerate points (tol="
                      << spatial_dedup_tol << ")\n";
    }

    for (int idx : to_remove) degenerate_pts.erase(idx);
    if (verbose)
        std::cout << "Filtered out " << to_remove.size()
                  << " degenerate points. Remaining: " << degenerate_pts.size() << "\n";
}

// ── init_gradients_by_collinear_pairs ───────────────────────────────

Eigen::MatrixXd init_gradients_by_collinear_pairs(
    const Eigen::MatrixXd& sdf_points,
    const Eigen::VectorXd& sdf_values,
    const Options& options,
    double ratio_tol)
{
    int N = (int)sdf_points.rows();
    // Accumulate gradient votes per point
    std::vector<Eigen::Vector3d> grad_sum(N, Eigen::Vector3d::Zero());
    std::vector<int> vote_count(N, 0);

    for (int i = 0; i < N; i++) {
        for (int j : options.ngbrs_list[i]) {
            if (j <= i) continue;
            Eigen::Vector3d diff = sdf_points.row(i) - sdf_points.row(j);
            double dist = diff.norm();
            if (dist < 1e-10) continue;
            double delta = sdf_values(i) - sdf_values(j);
            if (std::abs(std::abs(delta) / dist - 1.0) > ratio_tol) continue;
            Eigen::Vector3d g = diff * (delta >= 0 ? 1.0 : -1.0) / dist;
            grad_sum[i] += g;   vote_count[i]++;
            grad_sum[j] -= g;   vote_count[j]++;
        }
    }

    Eigen::MatrixXd grads = Eigen::MatrixXd::Constant(N, 3, std::numeric_limits<double>::quiet_NaN());
    int filled = 0;
    for (int i = 0; i < N; i++) {
        if (vote_count[i] == 0) continue;
        double norm = grad_sum[i].norm();
        if (norm < 1e-10) continue;
        grads.row(i) = (grad_sum[i] / norm).transpose();
        filled++;
    }
    if (options.verbose)
        std::cout << "[init_gradients_by_collinear_pairs] filled " << filled
                  << " / " << N << " gradients\n";
    return grads;
}

// ── init_gradients_by_degenerate_pts ────────────────────────────────

Eigen::MatrixXd init_gradients_by_degenerate_pts(
    const Eigen::MatrixXd& sdf_points,
    const Eigen::VectorXd& sdf_values,
    Interpolator& interpolator,
    Options& options)
{
    int N = (int)sdf_points.rows();
    auto& degenerate_pts = options.degenerate_pts;
    double t_fit_init = 0;  // accumulate the two out-of-loop RBF fits
    // 1. Initial fit without gradients
    auto _tf1 = clk::now();
    interpolator.fit(sdf_points, sdf_values);
    t_fit_init += ms_since(_tf1)/1000.0;
    if (options.verbose)
        std::cout << "======== first fit done with input " << N << " points\n";

    // Filter degenerate points. spatial_dedup_tol=0 disables Step 4
    // cross-sphere KD-tree dedup so the logged count reflects "spheres with
    // vanishing exposed region" before any spatial collapsing.
    filter_degenerate_pts(degenerate_pts, interpolator, options.verbose,
                          /*dist_tol=*/0.1, /*spatial_dedup_tol=*/0.0);

    // Append one row per run: "<grid_len> <remaining_degen> <fully_covered>".
    if (false) {
        std::ofstream f("logs/degen_stats.txt", std::ios::app);
        f << options.grid_len << " " << degenerate_pts.size()
          << " " << options.fully_covered << "\n";
    }
    

    if (options.export_short_arcs)
        export_short_arcs_ply(sdf_points, options, "out/" + options.name);

    // Initialize gradients to NaN
    Eigen::MatrixXd init_grads = Eigen::MatrixXd::Constant(N, 3, std::numeric_limits<double>::quiet_NaN());

    // Add degenerate points as zero-value constraints
    Eigen::MatrixXd to_train_points = sdf_points;
    Eigen::VectorXd to_train_sdf = sdf_values;

    std::vector<Eigen::Vector3d> pts_to_add;
    for (auto& [i, pts] : degenerate_pts) {
        pts_to_add.push_back(pts[0]);
        init_grads.row(i) = (sdf_points.row(i) - pts[0].transpose()) / (sdf_values(i) + 1e-10);
    }

    if (!pts_to_add.empty()) {
        int n_add = (int)pts_to_add.size();
        Eigen::MatrixXd new_pts(N + n_add, 3);
        new_pts.topRows(N) = sdf_points;
        for (int i = 0; i < n_add; i++)
            new_pts.row(N + i) = pts_to_add[i].transpose();

        Eigen::VectorXd new_vals(N + n_add);
        new_vals.head(N) = sdf_values;
        new_vals.tail(n_add).setZero();

        to_train_points = new_pts;
        to_train_sdf = new_vals;
    }

    if (options.verbose)
        std::cout << "After adding points for degenerate arcs, total points: "
                  << to_train_points.rows() << "\n";
    auto _tf2 = clk::now();
    interpolator.fit(to_train_points, to_train_sdf);
    t_fit_init += ms_since(_tf2)/1000.0;
    if (options.verbose) {
        std::cout << "======== second fit done with input " << to_train_points.rows()
                  << " points (including " << pts_to_add.size() << " degenerate arc points)\n";
        std::cout << "initial gradient estimation done\n";
        std::cout << "[DECOMP-init] rbf_fit: " << t_fit_init << " s\n";
    }

    // Debug check
    // for (auto& [i, pts] : degenerate_pts) {
    //     if ((int)pts.size() != 1) {
    //         std::cout << "ERRRRRRRRRRR\n";
    //     }
    //     init_grads.row(i) = (sdf_points.row(i) - pts[0].transpose()) / (sdf_values(i) + 1e-10);
    //     Eigen::Vector3d proj = sdf_points.row(i).transpose() - sdf_values(i) * init_grads.row(i).transpose();
    //     Eigen::MatrixXd proj_mat(1, 3);
    //     proj_mat.row(0) = proj.transpose();
    //     double pred_sdf = interpolator.predict(proj_mat)(0);
    //     if (pred_sdf > 0.6) {
    //         std::cout << "Warning: For degenerate point " << i
    //                   << ", the projected point is still outside with predicted sdf "
    //                   << pred_sdf << ".\n";
    //     }
    // }

    return init_grads;
}

// ── main_algorithm ──────────────────────────────────────────────────

MainResult main_algorithm(
    const Eigen::MatrixXd& sdf_points,
    const Eigen::VectorXd& sdf_values,
    Options& options)
{
    int N = (int)sdf_points.rows();

    auto t0 = clk::now();
    if (options.verbose)
        std::cout << "Starting main_algorithm with " << N << " points\n";

    // Step 1: Compute visible arcs + degenerate points
    auto t1 = clk::now();
    get_visible_arcs(sdf_points, sdf_values, options);
    if (options.turn_off_short_arcs)
        options.degenerate_pts.clear();
    if (options.verbose)
        std::cout << "[main_algorithm] get_visible_arcs: " << ms_since(t1)/1000.0 << " s\n";

    // Create interpolator
    auto t2 = clk::now();    
    // Fit phase has many small parallel sections — too many threads hurt.
    // Auto-detected on big boxes, no-op on Mac / small workstations.
    int _saved_threads = sdf::set_threads(sdf::thread_policy().fit);

    std::shared_ptr<Interpolator> interpolator;
    if (options.interpolator_type == "Duchon") {
        interpolator = std::make_shared<DuchonInterpolator>("cubic", options.reg);
    } else {
        if (options.verbose)
            std::cout << "Using regularization " << options.reg << " for PUInterpolator\n";
        interpolator = std::make_shared<PUInterpolator>(
            "cubic", options.interp_overlap,
            10, 200, options.reg, options.interp_partition,
            options.verbose, options.pair_local);
    }
    if (options.verbose)
        std::cout << "[main_algorithm] interpolator ctor: " << ms_since(t2)/1000.0 << " s\n";

    Eigen::MatrixXd gradients;
    if (options.gt_gradients) {
        // GT mode: skip init + iteration, fit once with ground-truth gradients
        auto tgt = clk::now();
        const Eigen::MatrixXd& gt = *options.gt_gradients;
        Eigen::VectorXi all_vis = Eigen::VectorXi::Ones(N);
        interpolator->fit(sdf_points, sdf_values, &gt, &all_vis);
        gradients = gt;
        if (options.verbose)
            std::cout << "[main_algorithm] gt_gradients mode, fit: "
                      << ms_since(tgt)/1000.0 << " s\n";
    } else {
        // Step 1b: Initial gradient estimation using degenerate points
        auto t3 = clk::now();
        Eigen::MatrixXd init_grads = init_gradients_by_degenerate_pts(
            sdf_points, sdf_values, *interpolator, options);
        if (options.verbose)
            std::cout << "[main_algorithm] init_gradients_by_degenerate_pts: "
                      << ms_since(t3)/1000.0 << " s\n";

        // Step 2: Iterative optimization
        auto t4 = clk::now();
        gradients = iterative_projection_3d(
            sdf_points, sdf_values, init_grads,
            *interpolator, options,
            options.max_iters,
            /*num_coarse=*/64,
            options.optim_steps,
            /*lr=*/options.lr);
        if (options.verbose)
            std::cout << "[main_algorithm] iterative_projection_3d: "
                      << ms_since(t4)/1000.0 << " s\n";
    }

    // Final projection + visibility check
    auto t5 = clk::now();
    Eigen::MatrixXd projections(N, 3);
    for (int i = 0; i < N; i++)
        projections.row(i) = sdf_points.row(i) - sdf_values(i) * gradients.row(i);

    std::vector<char> frozen(N, 0);
    for (const auto& [di, dpts] : options.degenerate_pts)
        if (di >= 0 && di < N) frozen[di] = 1;
    Eigen::VectorXi vis = are_points_visible(
        projections, sdf_values, frozen,
        options.ngbrs_list, *options.sphere_bvh, 1e-8, options.verbose);
    if (options.verbose) {
        std::cout << "[main_algorithm] final projection + visibility: "
                  << ms_since(t5)/1000.0 << " s\n";
        std::cout << "[main_algorithm] total: " << ms_since(t0)/1000.0 << " s\n";
    }

    if (options.export_projections)
        export_projection_ply(sdf_points, projections, vis, options, "out/" + options.name);

    sdf::restore_threads(_saved_threads);
    return {projections, vis, interpolator};
}

// ── export_short_arcs_ply ────────────────────────────────────────────

static void export_short_arcs_ply_old(
    const Eigen::MatrixXd& sdf_points,
    const Options& options,
    const std::string& out_dir)
{
    const auto& dpts = options.degenerate_pts;
    if (dpts.empty()) return;

    std::system(("mkdir -p " + out_dir).c_str());
    std::string path = out_dir + "/shortArcs_"
        + options.name + "_"
        + std::to_string(options.grid_len) + ".ply";

    // Flatten: one entry per (sphere, degenerate_pt) pair, sorted by sphere index
    struct Entry { int sphere_idx; Eigen::Vector3d surf_pt; };
    std::vector<Entry> entries;
    for (auto& [i, pts] : dpts)
        for (auto& p : pts)
            entries.push_back({i, p});
    std::sort(entries.begin(), entries.end(), [](auto& a, auto& b){ return a.sphere_idx < b.sphere_idx; });

    const int M = static_cast<int>(entries.size());
    std::ofstream f(path);
    if (!f) { std::cerr << "[export_short_arcs_ply] cannot open " << path << "\n"; return; }

    f << "ply\nformat ascii 1.0\n"
      << "element vertex " << 2 * M << "\n"
      << "property float x\nproperty float y\nproperty float z\n"
      << "property uchar red\nproperty uchar green\nproperty uchar blue\n"
      << "element edge " << M << "\n"
      << "property int vertex1\nproperty int vertex2\nend_header\n";

    // SDF sphere centers: gray (one per entry, may repeat for multi-point spheres)
    for (auto& e : entries)
        f << sdf_points(e.sphere_idx,0) << " " << sdf_points(e.sphere_idx,1)
          << " " << sdf_points(e.sphere_idx,2) << " 100 100 100\n";
    // Degenerate surface points: green
    for (auto& e : entries)
        f << e.surf_pt.x() << " " << e.surf_pt.y() << " " << e.surf_pt.z() << " 0 200 0\n";
    // Edges: sphere center k  <->  surface point M+k
    for (int k = 0; k < M; k++)
        f << k << " " << (M + k) << "\n";

    std::cout << "[export_short_arcs_ply] wrote " << path
              << "  (" << M << " arcs from " << dpts.size() << " spheres)\n";
}

static void export_short_arcs_ply_new(
    const Eigen::MatrixXd& sdf_points,
    const Options& options,
    const std::string& out_dir)
{
    const auto& dpts = options.degenerate_pts;
    if (dpts.empty()) return;

    std::system(("mkdir -p " + out_dir).c_str());

    // Flatten: one entry per (sphere, degenerate_pt) pair, sorted by sphere index
    struct Entry { int sphere_idx; Eigen::Vector3d surf_pt; };
    std::vector<Entry> entries;
    for (auto& [i, pts] : dpts)
        for (auto& p : pts)
            entries.push_back({i, p});
    std::sort(entries.begin(), entries.end(), [](auto& a, auto& b){ return a.sphere_idx < b.sphere_idx; });

    // Spatial dedup at export tol — different spheres' degenerate projections
    // can land on the same arc/edge. KDTree once + earlier-kept-wins sweep.
    constexpr double export_dedup_tol = 0;
    std::vector<int> kept;
    kept.reserve(entries.size());
    if (!entries.empty()) {
        Eigen::MatrixXd P((int)entries.size(), 3);
        for (int i = 0; i < (int)entries.size(); i++) P.row(i) = entries[i].surf_pt.transpose();
        KDTree3D tree(P);
        std::vector<char> alive(entries.size(), 1);
        for (int i = 0; i < (int)entries.size(); i++) {
            if (!alive[i]) continue;
            kept.push_back(i);
            for (int j : tree.query_ball_point(entries[i].surf_pt, export_dedup_tol)) {
                if (j > i) alive[j] = 0;
            }
        }
    }

    const int M = static_cast<int>(kept.size());
    std::string path = out_dir + "/shortArcs_"
        + options.name + "_"
        + std::to_string(options.grid_len) + "_"
        + std::to_string(M) + "pts.ply";
    std::ofstream f(path);
    if (!f) { std::cerr << "[export_short_arcs_ply] cannot open " << path << "\n"; return; }

    f << "ply\nformat ascii 1.0\n"
      << "element vertex " << M << "\n"
      << "property float x\nproperty float y\nproperty float z\n"
      << "property uchar red\nproperty uchar green\nproperty uchar blue\n"
      << "end_header\n";

    // // SDF sphere centers: gray (one per entry, may repeat for multi-point spheres)
    // for (int k : kept)
    //     f << sdf_points(entries[k].sphere_idx,0) << " "
    //       << sdf_points(entries[k].sphere_idx,1) << " "
    //       << sdf_points(entries[k].sphere_idx,2) << " 100 100 100\n";

    // Degenerate surface points: green
    for (int k : kept) {
        const auto& e = entries[k];
        f << e.surf_pt.x() << " " << e.surf_pt.y() << " " << e.surf_pt.z() << " 0 200 0\n";
    }
    // // Edges: sphere center k  <->  surface point M+k
    // for (int k = 0; k < M; k++)
    //     f << k << " " << (M + k) << "\n";

    std::cout << "[export_short_arcs_ply] wrote " << path
              << "  (" << M << " unique pts from " << entries.size()
              << " arcs across " << dpts.size() << " spheres)\n";
}

static void export_short_arcs_ply(
    const Eigen::MatrixXd& sdf_points,
    const Options& options,
    const std::string& out_dir,
    bool use_new)
{
    if (use_new) export_short_arcs_ply_new(sdf_points, options, out_dir);
    else         export_short_arcs_ply_old(sdf_points, options, out_dir);
}
// Helper: write one PLY file for a subset of points (selected by `mask`).
// Vertices are laid out as: [sdf_pts for mask] then [proj_pts for mask].
// Edges connect vertex i to vertex (count + i) for each selected index.
static void write_ply_subset(
    const Eigen::MatrixXd& sdf_points,
    const Eigen::MatrixXd& projections,
    const std::vector<int>& indices,
    int sdf_r, int sdf_g, int sdf_b,
    int proj_r, int proj_g, int proj_b,
    const std::string& path)
{
    const int M = static_cast<int>(indices.size());

    std::ofstream f(path);
    if (!f) {
        std::cerr << "[export_projection_ply] cannot open " << path << "\n";
        return;
    }

    f << "ply\n"
      << "format ascii 1.0\n"
      << "element vertex " << 2 * M << "\n"
      << "property float x\n"
      << "property float y\n"
      << "property float z\n"
      << "property uchar red\n"
      << "property uchar green\n"
      << "property uchar blue\n"
      << "element edge " << M << "\n"
      << "property int vertex1\n"
      << "property int vertex2\n"
      << "end_header\n";

    for (int i : indices)
        f << sdf_points(i,0) << " " << sdf_points(i,1) << " " << sdf_points(i,2)
          << " " << sdf_r << " " << sdf_g << " " << sdf_b << "\n";

    for (int i : indices)
        f << projections(i,0) << " " << projections(i,1) << " " << projections(i,2)
          << " " << proj_r << " " << proj_g << " " << proj_b << "\n";

    for (int k = 0; k < M; k++)
        f << k << " " << (M + k) << "\n";
}

void export_projection_ply(
    const Eigen::MatrixXd& sdf_points,
    const Eigen::MatrixXd& projections,
    const Eigen::VectorXi& vis,
    const Options& options,
    const std::string& out_dir)
{
    const int N = static_cast<int>(sdf_points.rows());

    std::system(("mkdir -p " + out_dir).c_str());

    std::string stem = out_dir + "/projection_"
        + options.name + "_"
        + std::to_string(options.grid_len) + "_"
        + std::to_string(options.max_iters);
    if (options.use_MES == -1) stem += "_noMES";
    else if (options.use_MES == 1) stem += "_MESforce";
    else stem += "_MES";

    std::vector<int> vis_idx, invis_idx;
    for (int i = 0; i < N; i++) {
        if (vis(i) != 0) vis_idx.push_back(i);
        else             invis_idx.push_back(i);
    }

    // File 1: visible pairs — gray SDF points, blue projection points
    write_ply_subset(sdf_points, projections, vis_idx,
                     100, 100, 100,   // SDF: gray
                     0, 0, 255,       // proj: blue
                     stem + ".ply");
    std::cout << "[export_projection_ply] " << stem << ".ply"
              << "  (" << vis_idx.size() << " visible pairs)\n";

    // File 2: invisible pairs — gray SDF points, red projection points
    write_ply_subset(sdf_points, projections, invis_idx,
                     100, 100, 100,   // SDF: gray
                     255, 50, 50,     // proj: red
                     stem + "_invisible.ply");
    std::cout << "[export_projection_ply] " << stem << "_invisible.ply"
              << "  (" << invis_idx.size() << " invisible pairs)\n";
}

}  // namespace sdf
