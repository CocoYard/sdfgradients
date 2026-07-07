#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <optional>
#include "main_algorithm.h"
#include "duchon_interpolator.h"
#include "pu_interpolator.h"
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
        .def_readwrite("lr", &sdf::Options::lr);

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
        .def("is_trained", &sdf::DuchonInterpolator::is_trained);

    py::class_<sdf::PUInterpolator, sdf::Interpolator, std::shared_ptr<sdf::PUInterpolator>>(m, "PUInterpolator")
           .def(py::init<const std::string&, double, int, int, double, const std::string&, bool, bool>(),
             py::arg("kernel") = "cubic", py::arg("overlap") = 0.25,
             py::arg("min_points") = 10, py::arg("max_points") = 200,
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
        .def("is_trained", &sdf::PUInterpolator::is_trained);

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
