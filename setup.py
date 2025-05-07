import os
import platform
import subprocess
import shutil
from setuptools import setup
from setuptools.command.build_py import build_py as _build_py

class BuildCPPProject(_build_py):
    def run(self):
        # Define the repository URL and local directory for the C++ project.
        repo_url = "https://github.com/YipengQin/VTP_source_code.git"
        base_dir = os.path.dirname(os.path.abspath(__file__))
        repo_dir = os.path.join(base_dir, "VTP_cpp")

        # --- Step 1. Clone the C++ repo if not already cloned ---
        if not os.path.exists(repo_dir):
            print("Cloning C++ repository...")
            subprocess.check_call(["git", "clone", repo_url, repo_dir])
        else:
            print("C++ repository already cloned.")

        # --- Step 2. Build the C++ project using CMake ---
        build_dir = os.path.join(repo_dir, "build")
        os.makedirs(build_dir, exist_ok=True)
        print("Configuring the C++ build with CMake...")
        subprocess.check_call(["cmake", ".."], cwd=build_dir)
        print("Building the C++ project...")
        subprocess.check_call(["cmake", "--build", "."], cwd=build_dir)

        # --- Step 3. Locate the compiled executable ---
        exe_name = "VTP.exe" if platform.system() == "Windows" else "VTP"
        exe_path = os.path.join(build_dir, exe_name)
        if not os.path.exists(exe_path):
            raise RuntimeError(f"Compilation failed: {exe_path} not found")
        print(f"Found executable: {exe_path}")

        # --- Step 4. Copy the executable into your package folder ---
        package_bin_dir = os.path.join(base_dir, "dpu_mini", "bin")
        os.makedirs(package_bin_dir, exist_ok=True)
        destination = os.path.join(package_bin_dir, exe_name)
        print(f"Copying executable to {destination}...")
        shutil.copy(exe_path, destination)

        # Continue with the normal build process.
        _build_py.run(self)

setup(
    # Since metadata is now in setup.cfg, we only need to reference it.
    cmdclass={"build_py": BuildCPPProject},
)
