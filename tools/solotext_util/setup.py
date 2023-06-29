import os
import sys
from glob import glob

from setuptools import setup, find_packages

import torch
from torch.utils.cpp_extension import CUDAExtension
from torch.utils.cpp_extension import BuildExtension


requirements = ["torch", "torchvision"]


def get_extensions():
    srcs = glob("solotext_util/src/*.cu")
    extra_compile_args = {
        "cxx": [],
        "nvcc": [
            # "-rdc=true",
            # "--cudadevrt=static",
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
    }

    CC = os.environ.get("CC", None)
    if CC is not None:
        extra_compile_args["nvcc"].append("-ccbin={}".format(CC))

    include_dirs = ["solotext_util/src"]

    ext_modules = [
        CUDAExtension(
            "solotext_util._C",
            srcs,
            include_dirs=include_dirs,
            define_macros=[],
            extra_compile_args=extra_compile_args
        )
    ]

    return ext_modules


setup(
    # Meta Data
    name='solotext_util',
    version='0.1',
    description='solotext_util',
    # Package Info
    zip_safe=False,
    packages=find_packages(),
    ext_modules=get_extensions(),
    cmdclass={
        "build_ext": BuildExtension.with_options(no_python_abi_suffix=True),
    },
)
