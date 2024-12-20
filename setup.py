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
    "cupy-cuda12x>=13.3.0",
    "flash-attn>=2.6.3",
    "torch>=2.4",
    "triton>=3.1.0",
]
setup_requires = [
    "ninja",
    "packaging",
    "psutil", 
]
tests_requires = ["pytest"]

setup(
    name="dram-attention",
    version=__version__,
    description="Attend to main memory (DRAM) instead of GPU memory (HBM/VRAM) for long-context self-attention.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Tanthong Nguyen",
    url=URL,
    keywords=[
        "dram",
        "hbm",
        "offload",
        "pytorch",
        "self-attention",
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
