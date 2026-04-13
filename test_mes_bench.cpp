// Mini reproducer to bench MES against the reference.
// Mirrors bench/empty_spheres_reconstruction_3.cpp from the fork so we can
// compile with our build_mes_contact.py flags and compare timing.
//
// Usage:  ./test_mes_bench /path/to/tmp.csv
// Reads (x,y,z,signed_radius) per line, prints timing.

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Maximal_empty_spheres/contact_points_from_signed_distances.h>
#include <CGAL/Timer.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::Point_3   Point;
typedef Kernel::Vector_3  Vector;
typedef Kernel::Sphere_3  Sphere;
typedef std::pair<Point, Vector> Point_with_normal;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <csv_path>\n";
        return 1;
    }

    std::ifstream in(argv[1]);
    std::vector<std::pair<Sphere,int>> input_spheres;
    std::vector<Point_with_normal> pwns;

    double x, y, z, r;
    while (in >> x) {
        in.ignore(10, ','); in >> y;
        in.ignore(10, ','); in >> z;
        in.ignore(10, ','); in >> r;
        int inside = (r < 0) ? -1 : 1;
        input_spheres.emplace_back(Sphere(Point(x,y,z), CGAL::square(r)), inside);
    }
    std::cout << "Read " << input_spheres.size() << " spheres" << std::endl;

    CGAL::Timer timer;
    timer.start();
    CGAL::contact_points_from_signed_distances(
        input_spheres, std::back_inserter(pwns),
        /*filter_contact_spheres_bbx=*/true, /*debug_level=*/1);
    std::cout << "Computed " << pwns.size()
              << " contact points with normals in " << timer.time() << " sec."
              << std::endl;

    return 0;
}
