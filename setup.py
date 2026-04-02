import os
from setuptools import setup, find_packages

def read_requirements():
    with open("requirements.txt") as f:
        return f.read().splitlines()
    
setup(
    name="dpu_mini",
    packages=find_packages(),
    install_requires=read_requirements()
)
base_dir = os.path.dirname(os.path.abspath(__file__))
cli_setup_file = os.path.join(base_dir, "cli_setup.sh")

print('********************************************************')
print('********************************************************')
print('********************************************************')
print('ADD THE FOLLOWING TO .bash_profile TO ENABLE CLI TOOLS!!!')
print(f'source {cli_setup_file}')
