"""Setup DRAM ATTENTION package."""

from setuptools import find_namespace_packages, setup


def _get_version():
    with open("dram_attention/__init__.py", encoding="utf-8") as file:
        for line in file:
            if line.startswith("__version__"):
                _globals = {}
                exec(line, _globals)  # pylint: disable=exec-used
                return _globals["__version__"]
        raise ValueError("`__version__` not defined in `dram_attention/__init__.py`")


__version__ = _get_version()
URL = "https://github.com/long-context/dram-attention"

install_requires = [
    "torch>=2.4",
    "cupy-cuda12x>=13.3.0",
]
setup_requires = []
tests_requires = [
    "pytest",
    "tqdm",
]

setup(
    name="dram-attention",
    version=__version__,
    description="Attend to main memory (DRAM) instead of GPU memory (HBM/VRAM) for long-context self-attention.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Tanthong Nguyen",
    url=URL,
    keywords=[
        "self-attention",
        "pytorch",
        "hbm",
        "dram",
        "offload",
    ],
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_requires,
    packages=find_namespace_packages(exclude=["examples", "tests"]),
    extras_require={"test": tests_requires},
    python_requires=">=3.10",
    include_package_data=True,
    zip_safe=False,
)
