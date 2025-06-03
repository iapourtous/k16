from setuptools import setup, find_packages

setup(
    name="k16",
    version="1.0.0",
    description="K16 - Fast Approximate Nearest Neighbor Search Library",
    author="Dess4ever",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "faiss-cpu>=1.7.4",
        "joblib>=1.2.0",
        "pyyaml>=6.0",
        "scikit-learn>=1.2.0",
        "tqdm>=4.64.0",
    ],
    extras_require={
        "kneed": ["kneed>=0.8.0"],
        "gpu": ["faiss-gpu>=1.7.4"],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)