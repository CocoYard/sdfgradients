#include "optimization.h"
#include "visibility.h"
#include "clamp.h"
#include "mes_contact_core.h"
#include "thread_policy.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <chrono>

namespace sdf {

Eigen::MatrixXd iterative_projection_3d(
    const Eigen::MatrixXd& points,
    const Eigen::VectorXd& values,
    const Eigen::MatrixXd& init_gradients,
    Interpolator& interpolator,
    Options& options,
    int num_iter,
    int num_coarse,
    int optim_steps,
    double lr,
    const Eigen::MatrixXd* gt_gradients)
{
    int N = (int)points.rows();
    Eigen::MatrixXd gradients = init_gradients;
    bool MES_used = false;
    Eigen::VectorXi vis_cached;
    bool vis_cache_valid = false;

    if (gt_gradients) {
        double cos_sum = 0.0;
        for (int i = 0; i < N; i++)
            cos_sum += gradients.row(i).dot(gt_gradients->row(i));
        std::cout << "cos_sim mean to the ground truth gradients: " << cos_sum / N << "\n";
    }

    for (int it = 0; it < num_iter; it++) {
        std::cout << "Iter " << (it + 1) << " | ";
        // ── Step 1: Find best gradient via angular search ──────────
        auto t_sbg0 = std::chrono::high_resolution_clock::now();
        Eigen::MatrixXd new_gradients;
        if (options.iter_gradient_finding == "optimize") {
            new_gradients = interpolator.optimize_best_gradients(
                points, values, num_coarse, optim_steps, lr, &gradients);
        } else {
            new_gradients = interpolator.sample_best_gradients(
                points, values, num_coarse, /*refine_steps=*/4, /*num_refine=*/5, &gradients);
        }
        std::cout << "[" << options.iter_gradient_finding << "_best_gradients] "
                  << std::chrono::duration<double>(
                         std::chrono::high_resolution_clock::now() - t_sbg0).count()
                  << "s\n";

        // ── Clamp to arcs ──────────────────────────────────────────
        // Throughput-bound (N × neighbors), so bump threads up to predict
        // count for the duration — the surrounding iterative_projection_3d
        // runs under fit's reduced thread pool.
        if (options.clamp) {
            int _saved = sdf::set_threads(sdf::thread_policy().predict);
            Eigen::MatrixXd pre_clamp_gradients = new_gradients;
            int clamped_cnt = 0;
            while (true) {
                new_gradients = pre_clamp_gradients;
                clamped_cnt = clamp_gradients_to_arcs(
                    points, values, new_gradients,
                    options.degenerate_pts, options.batch,
                    options.ngbrs_list, *options.sphere_bvh, options.tolerance);
                if (1. * clamped_cnt / N > .3) {
                    options.tolerance.clamp_sdf_tol /= 2;
                } else {
                    break;
                }
            }
            sdf::restore_threads(_saved);
        }

        // ── Visibility checks ──────────────────────────────────────
        Eigen::MatrixXd proj_new(N, 3);
        for (int i = 0; i < N; i++)
            proj_new.row(i) = points.row(i) - values(i) * new_gradients.row(i);
        std::cout << "Checking visibility ...\n";
        Eigen::VectorXi vis_new = are_points_visible(proj_new, values, options.degenerate_pts, options.ngbrs_list, *options.sphere_bvh);
        Eigen::VectorXi vis_old;
        if (vis_cache_valid) {
            vis_old = vis_cached;
        } else {
            Eigen::MatrixXd proj_old(N, 3);
            for (int i = 0; i < N; i++)
                proj_old.row(i) = points.row(i) - values(i) * gradients.row(i);
            vis_old = are_points_visible(proj_old, values, options.degenerate_pts, options.ngbrs_list, *options.sphere_bvh);
        }
        // Don't update gradients that would make visible projections invisible
        // Don't update degenerate-arc points
        Eigen::VectorXi vis_next(N);
        for (int i = 0; i < N; i++) {
            bool skip = (vis_old(i) && !vis_new(i));
            if (options.degenerate_pts.count(i)) skip = true;
            if (skip) new_gradients.row(i) = gradients.row(i);
            vis_next(i) = skip ? vis_old(i) : vis_new(i);
        }
        vis_cached = vis_next;
        vis_cache_valid = true;

        std::cout << "  visible old: " << vis_old.sum() << ", "
                  << "visible new: " << vis_new.sum() << std::endl;

        // Compute visible mask (union of old and new)
        Eigen::VectorXi vis_mask(N);
        int visible_num = 0;
        for (int i = 0; i < N; i++) {
            vis_mask(i) = (vis_new(i) || vis_old(i));
            // vis_mask(i) = (vis_new(i) || vis_old(i)) && !options.ngbrs_list[i].empty();
            // vis_mask(i) = (vis_new(i) || vis_old(i)) && std::abs(values[i]) > 1e-2;
            if (vis_mask(i)) visible_num++;
        }
        std::cout << "Iter " << (it + 1) << " | ";
        std::cout << "Number of visible projected points: " << visible_num
                  << " out of " << N << ". Percentage: "
                  << (100.0 * visible_num / N) << "%\n";

        // ── MES contact points: when visibility improvement stalls ─
        // Mirrors optimization.py:589-594. Trigger once when new-vs-old
        // visibility gain < 1% of N. For points that are still invisible
        // and have a valid MES normal, override new_gradients with that
        // normal. vis_mask above is the union of old/new visibility.
        if (options.use_MES) {
            int vis_old_sum = vis_old.sum();
            double gain = (double)(visible_num - vis_old_sum) / N;
            if (!MES_used && gain < 0.01) {
                std::cout << "========= Using MES points... =========\n";
                auto mes_t0 = std::chrono::steady_clock::now();
                Eigen::MatrixXd contact_pts, mes_normals;
                mes_contact_core::contact_points_from_sdf(
                    points, values, /*filter_bbox=*/true, /*debug_level=*/0,
                    contact_pts, mes_normals);
                for (int i = 0; i < N; i++) {
                    bool valid = !std::isnan(mes_normals(i, 0));
                    bool not_visible = !vis_mask(i);
                    if (valid && not_visible)
                        new_gradients.row(i) = mes_normals.row(i);
                }
                auto mes_t1 = std::chrono::steady_clock::now();
                double mes_ms = std::chrono::duration<double, std::milli>(mes_t1 - mes_t0).count();
                std::cout << "MES contact points elapsed: " << mes_ms/1000 << " s\n";

                MES_used = true;
                vis_cache_valid = false;  // MES rewrote some gradients; cache stale
            }
        }

        gradients = new_gradients;

        // ── Step 2: Refit interpolant with current gradients ───────
        interpolator.fit(points, values, &gradients, &vis_mask);

        if (gt_gradients) {
            double gt_cos = 0.0;
            for (int i = 0; i < N; i++)
                gt_cos += gradients.row(i).dot(gt_gradients->row(i));
            std::cout << "cos_sim mean to the ground truth gradients: " << gt_cos / N << "\n";
        }
    }

    return gradients;
}

}  // namespace sdf
