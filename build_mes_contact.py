"""
Build mes_contact.cpython-*.so in the sdfgradients directory.
Run from sdfgradients/:  python3 build_mes_contact.py
"""
import os
import subprocess
import sys

CGAL_BASE = "/Users/yongcheng/Documents/phd/research/sdf/maximal-empty-spheres/cgal"

# Collect -isystem flags for every CGAL module that has an include/ dir
cgal_modules = os.path.join(CGAL_BASE, "cgal")
isystem_flags = []
for name in sorted(os.listdir(cgal_modules)):
    inc = os.path.join(cgal_modules, name, "include")
    if os.path.isdir(inc):
        isystem_flags.append(f"-isystem{inc}")

# pybind11 / numpy include paths (from the active venv)
pybind_includes = subprocess.check_output(
    [sys.executable, "-m", "pybind11", "--includes"], text=True
).split()

# Extension suffix (e.g. .cpython-312-darwin.so)
import sysconfig
ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")

src_dir = os.path.dirname(os.path.abspath(__file__))
src     = os.path.join(src_dir, "mes_contact.cpp")
out     = os.path.join(src_dir, f"mes_contact{ext_suffix}")

cmd = [
    "/usr/bin/c++",
    "-O3", "-DNDEBUG", "-std=gnu++17", "-arch", "arm64",
    "-shared", "-fPIC", "-undefined", "dynamic_lookup",
    "-DCGAL_USE_GMPXX=1",
    f"-I{CGAL_BASE}/build/_deps/eigen-src",
] + isystem_flags + [
    "-isystem/opt/homebrew/include",
] + pybind_includes + [
    "-Wl,-search_paths_first", "-Wl,-headerpad_max_install_names",
    src, "-o", out,
    "/opt/homebrew/lib/libgmpxx.dylib",
    "/opt/homebrew/lib/libmpfr.dylib",
    "/opt/homebrew/lib/libgmp.dylib",
]

print(f"Compiling {src} -> {out}")
result = subprocess.run(cmd, capture_output=True, text=True)
if result.stdout:
    print(result.stdout)
if result.stderr:
    print(result.stderr, file=sys.stderr)
if result.returncode == 0:
    print("Done.")
else:
    sys.exit(result.returncode)
