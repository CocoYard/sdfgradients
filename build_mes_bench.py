"""
Build test_mes_bench executable using the same flags as build_mes_contact.py
(minus the pybind/shared-lib bits). Lets us bench pure CGAL MES performance
isolated from pybind11.
"""
import os
import subprocess
import sys

CGAL_BASE = "/Users/yongcheng/Documents/phd/research/sdf/maximal-empty-spheres/cgal"

cgal_modules = os.path.join(CGAL_BASE, "cgal")
isystem_flags = []
for name in sorted(os.listdir(cgal_modules)):
    inc = os.path.join(cgal_modules, name, "include")
    if os.path.isdir(inc):
        isystem_flags.append(f"-isystem{inc}")

src_dir = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(src_dir, "test_mes_bench.cpp")
out = os.path.join(src_dir, "test_mes_bench")

cmd = [
    "/usr/bin/c++",
    "-O3", "-DNDEBUG", "-std=gnu++17", "-arch", "arm64",
    "-DCGAL_USE_GMPXX=1",
    f"-I{CGAL_BASE}/build/_deps/eigen-src",
] + isystem_flags + [
    "-isystem/opt/homebrew/include",
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
sys.exit(result.returncode)
