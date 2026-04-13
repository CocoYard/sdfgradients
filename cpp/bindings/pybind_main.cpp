#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <optional>
#include "main_algorithm.h"
#include "duchon_interpolator.h"
#include "pu_interpolator.h"
#include "visibility.h"

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
        .def_readwrite("grid_len", &sdf::Options::grid_len)
        .def_readwrite("max_iters", &sdf::Options::max_iters)
        .def_readwrite("clamp", &sdf::Options::clamp)
        .def_readwrite("turn_off_short_arcs", &sdf::Options::turn_off_short_arcs)
        .def_readwrite("use_MES", &sdf::Options::use_MES)
        .def_readwrite("reg", &sdf::Options::reg)
        .def_readwrite("interpolator_type", &sdf::Options::interpolator_type)
        .def_readwrite("interp_partition", &sdf::Options::interp_partition)
        .def_readwrite("interp_overlap", &sdf::Options::interp_overlap)
        .def_readwrite("tolerance", &sdf::Options::tolerance);

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
                                      int chunk_size) {
        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        self.extract_surface(bbox_min, bbox_max, nx, ny, nz, iso, V, F, chunk_size);
        return py::make_tuple(V, F);
    };

    py::class_<sdf::Interpolator, std::shared_ptr<sdf::Interpolator>>(m, "Interpolator")
        .def("predict", &sdf::Interpolator::predict,
             py::arg("x_new"), py::arg("chunk_size") = 5000)
        .def("predict_gradients", &sdf::Interpolator::predict_gradients,
             py::arg("x_new"), py::arg("chunk_size") = 5000)
        .def("extract_surface", extract_surface_wrapper,
             py::arg("bbox_min"), py::arg("bbox_max"),
             py::arg("nx"), py::arg("ny"), py::arg("nz"),
             py::arg("iso") = 0.0, py::arg("chunk_size") = 5000,
             "Extract an isosurface via libigl marching cubes. "
             "Returns (V, F) as numpy arrays.");

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
           .def(py::init<const std::string&, double, int, int, double, const std::string&>(),
             py::arg("kernel") = "cubic", py::arg("overlap") = 0.25,
             py::arg("min_points") = 10, py::arg("max_points") = 200,
               py::arg("reg") = 1e-5,
             py::arg("partition") = "box")
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
}
