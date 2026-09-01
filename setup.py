from setuptools import setup, find_packages

setup(
    name="ameva-vulkan-runtime",
    version="1.1.0",
    description="Unified Cross-Modal Vulkan GPU Acceleration Runtime & HAL for Mobile Android",
    long_description=open("README.pypi.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Eunho Kim",
    author_email="contact@uno-km.com",
    url="https://uno-km.vercel.app/lib/vulkan/",
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "ameva-gpu=ameva_vulkan_runtime.cli:main",
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
