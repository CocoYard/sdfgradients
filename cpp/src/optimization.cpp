#include "optimization.h"
#include "visibility.h"
#include "clamp.h"
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
    int refine_steps,
    int num_refine,
    const Eigen::MatrixXd* gt_gradients)
{
    int N = (int)points.rows();
    Eigen::MatrixXd gradients = init_gradients;

    if (gt_gradients) {
        double cos_sum = 0.0;
        for (int i = 0; i < N; i++)
            cos_sum += gradients.row(i).dot(gt_gradients->row(i));
        std::cout << "cos_sim mean to the ground truth gradients: " << cos_sum / N << "\n";
    }

    for (int it = 0; it < num_iter; it++) {
        // ── Step 1: Find best gradient via angular search ──────────
        auto t_sbg0 = std::chrono::high_resolution_clock::now();
        Eigen::MatrixXd new_gradients = interpolator.sample_best_gradients(
            points, values, num_coarse, refine_steps, num_refine, &gradients);
        std::cout << "  [sample_best_gradients] "
                  << std::chrono::duration<double>(
                         std::chrono::high_resolution_clock::now() - t_sbg0).count()
                  << "s\n";

        // ── Clamp to arcs ──────────────────────────────────────────
        if (options.clamp) {
            Eigen::MatrixXd pre_clamp_gradients = new_gradients;
            int clamped_cnt = 0;
            while (true) {
                new_gradients = pre_clamp_gradients;
                clamped_cnt = clamp_gradients_to_arcs(
                    points, values, new_gradients,
                    options.degenerate_pts, options.batch,
                    options.ngbrs_list, options.tolerance);
                if (1. * clamped_cnt / N > .3) {
                    options.tolerance.clamp_sdf_tol /= 2;
                } else {
                    break;
                }
            }
        }

        // ── Visibility checks ──────────────────────────────────────
        Eigen::MatrixXd proj_new(N, 3), proj_old(N, 3);
        for (int i = 0; i < N; i++) {
            proj_new.row(i) = points.row(i) - values(i) * new_gradients.row(i);
            proj_old.row(i) = points.row(i) - values(i) * gradients.row(i);
        }
        std::cout << "Checking visibility of new and old projections...\n";
        Eigen::VectorXi vis_new = are_points_visible(proj_new, points, values, options.ngbrs_list);
        Eigen::VectorXi vis_old = are_points_visible(proj_old, points, values, options.ngbrs_list);

        // Don't update gradients that would make visible projections invisible
        // Don't update degenerate-arc points
        for (int i = 0; i < N; i++) {
            bool skip = (vis_old(i) && !vis_new(i));
            if (options.degenerate_pts.count(i)) skip = true;
            if (skip) new_gradients.row(i) = gradients.row(i);
        }

        // ── Convergence diagnostic ─────────────────────────────────
        double cos_sum = 0.0;
        double min_cos = 1.0;
        for (int i = 0; i < N; i++) {
            double cs = gradients.row(i).dot(new_gradients.row(i));
            cos_sum += cs;
            min_cos = std::min(min_cos, cs);
        }
        double mean_cos = cos_sum / N;
        double max_angle_deg = std::acos(std::clamp(min_cos, -1.0, 1.0)) * 180.0 / M_PI;

        std::cout << "Iter " << (it + 1) << " | mean cos_sim: " << mean_cos
                  << "  max angle change: " << max_angle_deg << "°\n";

        // Compute visible mask (union of old and new)
        Eigen::VectorXi vis_mask(N);
        int visible_num = 0;
        for (int i = 0; i < N; i++) {
            vis_mask(i) = (vis_new(i) || vis_old(i)) && !options.ngbrs_list[i].empty();
            if (vis_mask(i)) visible_num++;
        }
        std::cout << "Number of visible projected points: " << visible_num
                  << " out of " << N << ". Percentage: "
                  << (100.0 * visible_num / N) << "%\n";

        // TODO: MES contact points integration
        // In the Python version, when visibility improvement stalls,
        // mes_contact.contact_points_from_sdf is called.
        // This requires linking to the existing mes_contact C++ module.
        // For now, this step is omitted.

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
