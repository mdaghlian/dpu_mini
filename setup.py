from setuptools import setup, find_packages

def read_requirements():
    with open("requirements.txt") as f:
        return f.read().splitlines()
    
setup(
    name="dpu_mini",
    packages=find_packages(include=['dpu_mini', 'dpu_mini.*']),
    install_requires=read_requirements()
)
