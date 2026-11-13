import os
import platform
import subprocess
import shutil
from setuptools import setup
from setuptools.command.build_py import build_py as _build_py

# The VTP exact-geodesic tool (C++, built with CMake) is optional: it is only
# needed for Surface.exact_geodesic_distance(). Opt in with
#     DPU_MINI_BUILD_VTP=1 pip install .
# A plain `pip install` needs no git, CMake, compiler or network access.
BUILD_VTP = os.environ.get("DPU_MINI_BUILD_VTP", "0").lower() in ("1", "true", "yes")


def build_vtp(base_dir):
    repo_url = "https://github.com/YipengQin/VTP_source_code.git"
    repo_dir = os.path.join(base_dir, "VTP_cpp")

    if not os.path.exists(repo_dir):
        print("Cloning VTP repository...")
        subprocess.check_call(["git", "clone", repo_url, repo_dir])

    build_dir = os.path.join(repo_dir, "build")
    os.makedirs(build_dir, exist_ok=True)
    subprocess.check_call(["cmake", ".."], cwd=build_dir)
    subprocess.check_call(["cmake", "--build", "."], cwd=build_dir)

    exe_name = "VTP.exe" if platform.system() == "Windows" else "VTP"
    exe_path = os.path.join(build_dir, exe_name)
    if not os.path.exists(exe_path):
        raise RuntimeError(f"VTP compilation failed: {exe_path} not found")

    package_bin_dir = os.path.join(base_dir, "dpu_mini", "bin")
    os.makedirs(package_bin_dir, exist_ok=True)
    shutil.copy(exe_path, os.path.join(package_bin_dir, exe_name))
    print(f"Installed VTP to {package_bin_dir}")


class BuildPy(_build_py):
    def run(self):
        if BUILD_VTP:
            build_vtp(os.path.dirname(os.path.abspath(__file__)))
        else:
            print("Skipping optional VTP build (set DPU_MINI_BUILD_VTP=1 to enable).")
        _build_py.run(self)


setup(cmdclass={"build_py": BuildPy})
