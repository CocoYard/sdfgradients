#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <optional>
#include "main_algorithm.h"
#include "duchon_interpolator.h"
#include "pu_interpolator.h"
#include "vsk_interpolator.h"
#include "visibility.h"
#include "mes_contact_core.h"

namespace py = pybind11;

PYBIND11_MODULE(sdf_cpp, m) {
    m.doc() = "C++ implementation of SDF gradient optimization";

    // ── Tolerance ───────────────────────────────────────────────────
    py::class_<sdf::Tolerance>(m, "Tolerance")
        .def(py::init<>())
        .def_readwrite("clamp_radius_ratio", &sdf::Tolerance::clamp_radius_ratio)
        .def_readwrite("clamp_sdf_tol", &sdf::Tolerance::clamp_sdf_tol)
        .def_readwrite("float_tol", &sdf::Tolerance::float_tol)
        .def_readwrite("angle_tol", &sdf::Tolerance::angle_tol);

    // ── Options ─────────────────────────────────────────────────────
    py::class_<sdf::Options>(m, "Options")
        .def(py::init<>())
        .def_readwrite("name", &sdf::Options::name)
        .def_readwrite("grid_len", &sdf::Options::grid_len)
        .def_readwrite("max_iters", &sdf::Options::max_iters)
        .def_readwrite("clamp", &sdf::Options::clamp)
        .def_readwrite("turn_off_short_arcs", &sdf::Options::turn_off_short_arcs)
        .def_readwrite("use_MES", &sdf::Options::use_MES)
        .def_readwrite("export_projections", &sdf::Options::export_projections)
        .def_readwrite("export_short_arcs",  &sdf::Options::export_short_arcs)
        .def_readwrite("verbose", &sdf::Options::verbose)
        .def_readwrite("reg", &sdf::Options::reg)
        .def_readwrite("interpolator_type", &sdf::Options::interpolator_type)
        .def_readwrite("interp_partition", &sdf::Options::interp_partition)
        .def_readwrite("interp_overlap", &sdf::Options::interp_overlap)
        .def_readwrite("pair_local", &sdf::Options::pair_local)
        .def_readwrite("tolerance", &sdf::Options::tolerance)
        .def_readwrite("gt_gradients", &sdf::Options::gt_gradients)
        .def_readwrite("iter_gradient_finding", &sdf::Options::iter_gradient_finding)
        .def_readwrite("grad_optimizer", &sdf::Options::grad_optimizer)
        .def_readwrite("lr", &sdf::Options::lr)
        .def_readwrite("optim_steps", &sdf::Options::optim_steps)
        .def_readwrite("degen_tol", &sdf::Options::degen_tol)
        // The short-arc candidates that survived filter_degenerate_pts, i.e.
        // exactly the zero-valued points the second RBF fit was given. Only
        // populated after main_algorithm has run on this Options object.
        .def_property_readonly("degenerate_points", [](const sdf::Options& o) {
            std::vector<int> idx;
            idx.reserve(o.degenerate_pts.size());
            for (const auto& kv : o.degenerate_pts)
                if (!kv.second.empty()) idx.push_back(kv.first);
            std::sort(idx.begin(), idx.end());  // hash order is not reproducible
            Eigen::VectorXi I((int)idx.size());
            Eigen::MatrixXd P((int)idx.size(), 3);
            for (int i = 0; i < (int)idx.size(); i++) {
                I(i) = idx[i];
                P.row(i) = o.degenerate_pts.at(idx[i])[0].transpose();
            }
            return py::make_tuple(I, P);
        }, "(indices, positions) of the surviving short-arc candidates.")
        // Every short-arc midpoint compute_exposed_batch produced, before
        // filter_degenerate_pts ran: one row per candidate, a sphere may own
        // several. Also only populated after main_algorithm.
        .def_property_readonly("short_arc_candidates", [](const sdf::Options& o) {
            const auto& si = o.batch.point_sphere_idx;
            Eigen::VectorXi I((int)si.size());
            for (int i = 0; i < (int)si.size(); i++) I(i) = si[i];
            return py::make_tuple(I, o.batch.point_positions);
        }, "(sphere_idx, positions) of every short-arc midpoint, pre-filter.");

    // ── MainResult ──────────────────────────────────────────────────
    py::class_<sdf::MainResult>(m, "MainResult")
        .def_readonly("projections", &sdf::MainResult::projections)
        .def_readonly("visibility_mask", &sdf::MainResult::visibility_mask)
        .def_readonly("interpolator", &sdf::MainResult::interpolator);

    // ── Interpolators ───────────────────────────────────────────────
    auto extract_surface_wrapper = [](const sdf::Interpolator& self,
                                      const Eigen::Vector3d& bbox_min,
                                      const Eigen::Vector3d& bbox_max,
                                      int nx, int ny, int nz,
                                      double iso,
                                      int chunk_size,
                                      bool lipschitz_postfix,
                                      bool use_dual_contouring) {
        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        self.extract_surface(bbox_min, bbox_max, nx, ny, nz, iso, V, F,
                             chunk_size, lipschitz_postfix,
                             use_dual_contouring);
        return py::make_tuple(V, F);
    };

    py::class_<sdf::Interpolator, std::shared_ptr<sdf::Interpolator>>(m, "Interpolator")
        .def("predict", &sdf::Interpolator::predict,
             py::arg("x_new"), py::arg("chunk_size") = 5000)
        .def("predict_gradients", &sdf::Interpolator::predict_gradients,
             py::arg("x_new"), py::arg("chunk_size") = 5000)
        .def_property("verbose",
             &sdf::Interpolator::verbose,
             &sdf::Interpolator::set_verbose)
        .def("extract_surface", extract_surface_wrapper,
             py::arg("bbox_min"), py::arg("bbox_max"),
             py::arg("nx"), py::arg("ny"), py::arg("nz"),
             py::arg("iso") = 0.0, py::arg("chunk_size") = 5000,
             py::arg("lipschitz_postfix") = true,
             py::arg("use_dual_contouring") = false,
             "Extract an isosurface from the implicit field. "
             "If use_dual_contouring=True, uses gradient-aware Dual Contouring "
             "(QEF per cell, sharp-feature preserving); otherwise uses libigl "
             "marching cubes. Returns (V, F) as numpy arrays.");

    // Helper lambda for fit() with optional pointer args
    auto fit_wrapper = [](sdf::Interpolator& self,
                          const Eigen::MatrixXd& points,
                          const Eigen::VectorXd& values,
                          std::optional<Eigen::MatrixXd> gradients,
                          std::optional<Eigen::VectorXi> mask) {
        const Eigen::MatrixXd* g = gradients ? &*gradients : nullptr;
        const Eigen::VectorXi* m = mask ? &*mask : nullptr;
        self.fit(points, values, g, m);
    };

    py::class_<sdf::DuchonInterpolator, sdf::Interpolator, std::shared_ptr<sdf::DuchonInterpolator>>(m, "DuchonInterpolator")
        .def(py::init<const std::string&>(), py::arg("kernel") = "cubic")
        .def("fit", [&fit_wrapper](sdf::DuchonInterpolator& self,
                                    const Eigen::MatrixXd& points,
                                    const Eigen::VectorXd& values,
                                    std::optional<Eigen::MatrixXd> gradients,
                                    std::optional<Eigen::VectorXi> mask) {
            fit_wrapper(self, points, values, gradients, mask);
        }, py::arg("points"), py::arg("values"),
           py::arg("gradients") = py::none(), py::arg("mask") = py::none())
        .def("predict", &sdf::DuchonInterpolator::predict,
             py::arg("x_new"), py::arg("chunk_size") = 500)
        .def("predict_gradients", &sdf::DuchonInterpolator::predict_gradients,
             py::arg("x_new"), py::arg("chunk_size") = 500)
        .def("set_normals", &sdf::DuchonInterpolator::set_normals, py::arg("N"),
             "Normals for the next fit(), one row per point; NaN rows are value-only.")
        .def("is_trained", &sdf::DuchonInterpolator::is_trained);

    py::class_<sdf::PUInterpolator, sdf::Interpolator, std::shared_ptr<sdf::PUInterpolator>>(m, "PUInterpolator")
           .def(py::init<const std::string&, double, double, const std::string&, bool, bool>(),
             py::arg("kernel") = "cubic", py::arg("overlap") = 0.25,
               py::arg("reg") = 1e-5,
             py::arg("partition") = "box",
             py::arg("verbose") = true,
             py::arg("pair_local") = true)
        .def("fit", [&fit_wrapper](sdf::PUInterpolator& self,
                                    const Eigen::MatrixXd& points,
                                    const Eigen::VectorXd& values,
                                    std::optional<Eigen::MatrixXd> gradients,
                                    std::optional<Eigen::VectorXi> mask) {
            fit_wrapper(self, points, values, gradients, mask);
        }, py::arg("points"), py::arg("values"),
           py::arg("gradients") = py::none(), py::arg("mask") = py::none())
        .def("predict", &sdf::PUInterpolator::predict,
             py::arg("x_new"), py::arg("chunk_size") = 5000)
        .def("predict_gradients", &sdf::PUInterpolator::predict_gradients,
             py::arg("x_new"), py::arg("chunk_size") = 5000)
        .def("set_normals", &sdf::PUInterpolator::set_normals, py::arg("N"),
             "Normals for the next fit(), one row per point; NaN rows are value-only.")
        .def("is_trained", &sdf::PUInterpolator::is_trained);

    py::class_<sdf::VSKInterpolator, sdf::PUInterpolator, std::shared_ptr<sdf::VSKInterpolator>>(m, "VSKInterpolator")
        .def(py::init<std::shared_ptr<sdf::Interpolator>, const std::string&, double, double,
                      const std::string&, bool, bool>(),
             py::arg("psi"), py::arg("kernel") = "cubic", py::arg("overlap") = 0.25,
             py::arg("reg") = 1e-5, py::arg("partition") = "box",
             py::arg("verbose") = true, py::arg("pair_local") = true,
             "PU interpolator whose local solves run on the lifted points (x, psi(x)); "
             "psi is any fitted interpolator.");

    // ── Functions ───────────────────────────────────────────────────
    m.def("are_points_visible",
          static_cast<Eigen::VectorXi (*)(
              const Eigen::MatrixXd&, const Eigen::MatrixXd&,
              const Eigen::VectorXd&, double)>(&sdf::are_points_visible),
          py::arg("query_points"), py::arg("sdf_points"),
          py::arg("sdf_values"), py::arg("epsilon") = 1e-8);

    m.def("main_algorithm", &sdf::main_algorithm,
          py::arg("sdf_points"), py::arg("sdf_values"), py::arg("options"),
          "Run the full SDF gradient optimization pipeline.\n"
          "Returns MainResult with projections and visibility_mask.");

    // ── Exposed-region arcs (debug / visualization) ──────────────────
    // Runs step 1 of the pipeline only — power-diagram neighbors plus
    // compute_exposed_batch — and hands the raw arc/cap geometry to Python.
    // The bench module (bench/sphere_exposed_pybind) can compute the same
    // thing, but only from the spatial-hash neighbor list, which is NOT the
    // power-cell facet set the carve needs; use this instead.
    m.def("exposed_arcs",
          [](const Eigen::MatrixXd& sdf_points, const Eigen::VectorXd& sdf_values,
             double degen_tol, bool verbose) {
              sdf::Options o;
              o.degen_tol = degen_tol;
              o.verbose = verbose;
              o.clamp = true;                  // keep the arc extraction path
              o.turn_off_short_arcs = false;
              sdf::get_visible_arcs(sdf_points, sdf_values, o);

              const auto& b = o.batch;
              const int n_caps_total = (int)b.cap_sphere_idx.size();

              // Caps are appended sphere by sphere in remap (compacted id)
              // order, so the id within each sphere's group is just its
              // running position — recovered here so Python can resolve
              // arc_cap_idx without assuming the layout.
              Eigen::VectorXi cap_id(n_caps_total);
              {
                  int run = 0, prev = -1;
                  for (int k = 0; k < n_caps_total; k++) {
                      int s = b.cap_sphere_idx[k];
                      run = (s == prev) ? run + 1 : 0;
                      prev = s;
                      cap_id(k) = run;
                  }
              }

              auto to_veci = [](const std::vector<int>& v) {
                  Eigen::VectorXi e((int)v.size());
                  for (int k = 0; k < (int)v.size(); k++) e(k) = v[k];
                  return e;
              };
              auto to_vecd = [](const std::vector<double>& v) {
                  Eigen::VectorXd e((int)v.size());
                  for (int k = 0; k < (int)v.size(); k++) e(k) = v[k];
                  return e;
              };

              Eigen::VectorXi n_caps((int)o.ngbrs_list.size());
              Eigen::VectorXi n_nbrs((int)o.ngbrs_list.size());
              n_caps.setZero();
              for (int k = 0; k < n_caps_total; k++) n_caps(b.cap_sphere_idx[k])++;
              for (int i = 0; i < (int)o.ngbrs_list.size(); i++)
                  n_nbrs(i) = (int)o.ngbrs_list[i].size();

              using namespace pybind11::literals;
              return py::dict(
                  "n_arcs"_a           = to_veci(b.n_arcs),
                  "n_points"_a         = to_veci(b.n_points),
                  "n_caps"_a           = n_caps,
                  "n_neighbors"_a      = n_nbrs,
                  "arc_sphere_idx"_a   = to_veci(b.arc_sphere_idx),
                  "arc_cap_idx"_a      = to_veci(b.arc_cap_idx),
                  "arc_start"_a        = to_vecd(b.arc_start),
                  "arc_end"_a          = to_vecd(b.arc_end),
                  "cap_sphere_idx"_a   = to_veci(b.cap_sphere_idx),
                  "cap_id"_a           = cap_id,
                  "cap_normals"_a      = b.cap_normals,
                  "cap_d"_a            = b.cap_d,
                  "cap_centers"_a      = b.cap_centers,
                  "cap_radii"_a        = b.cap_radii,
                  "cap_u"_a            = b.cap_u,
                  "cap_v"_a            = b.cap_v,
                  "point_sphere_idx"_a = to_veci(b.point_sphere_idx),
                  "point_positions"_a  = b.point_positions,
                  "fully_covered"_a    = o.fully_covered,
                  "hidden_points"_a    = o.hidden_points);
          },
          py::arg("sdf_points"), py::arg("sdf_values"),
          py::arg("degen_tol") = 1e-5, py::arg("verbose") = false,
          "Exposed-region arcs/caps/degenerate points for every SDF sphere,\n"
          "using the same power-diagram neighbors as the pipeline.\n"
          "Returns a dict of flat arrays (see BatchData in cpp/include/types.h).");

    // ── MES contact points / empty spheres ───────────────────────────
    m.def("contact_points_from_sdf",
          [](const Eigen::MatrixXd& points, const Eigen::VectorXd& sdf_values,
             bool filter_bbox, int debug_level) {
              Eigen::MatrixXd out_pts, out_normals, out_spheres;
              mes_contact_core::contact_points_from_sdf(
                  points, sdf_values, filter_bbox, debug_level,
                  out_pts, out_normals, &out_spheres);
              return py::make_tuple(out_pts, out_normals, out_spheres);
          },
          py::arg("points"), py::arg("sdf_values"),
          py::arg("filter_bbox") = true, py::arg("debug_level") = 0,
          "Compute maximal-empty-sphere contact points/normals for each SDF sample.\n"
          "Returns (contact_pts, normals, spheres). contact_pts/normals are (N,3),\n"
          "NaN rows where no contact sphere was found. spheres is (M,4): x,y,z,radius,\n"
          "radius > 0 for spheres built from outside (sdf>=0) samples, < 0 for inside.");
}
