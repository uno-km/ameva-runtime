import os
from setuptools import setup, find_packages

version_path = os.path.join(os.path.dirname(__file__), "python", "ameva_runtime", "_version.py")
version_ns = {}
if os.path.exists(version_path):
    with open(version_path, encoding="utf-8") as f:
        exec(f.read(), version_ns)
version = version_ns.get("__version__", "2.0.0")

setup(
    name="ameva-runtime",
    version=version,
    description="Unified Next-Gen Hardware Orchestration & AI Acceleration Runtime for Mobile & Edge",
    long_description=open("README.pypi.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Eunho Kim",
    author_email="contact@uno-km.com",
    url="https://github.com/uno-km/ameva-runtime",
    package_dir={"": "python"},
    packages=find_packages(where="python", exclude=["tests*", "*tests*"]),
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "ameva=ameva_runtime.cli:main",
            "ameva-run=ameva_runtime.cli:main",
            "ameva-gpu=ameva_runtime.cli:legacy_gpu_main",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Android",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
