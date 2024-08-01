#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
import glob
import os
from os import path
from setuptools import find_packages, setup
import torch
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

HAS_CUDA = torch.cuda.is_available() and CUDA_HOME is not None
FORCE_CUDA = os.getenv("FORCE_CUDA", "0") == "1"

LIBRARY_NAME = "trainingapi"

def get_extensions():
    this_dir = path.dirname(path.abspath(__file__))
    extensions_dir = path.join(this_dir, "src", "trainingapi", "model", "csrc")

    main_source = path.join(extensions_dir, "vision.cpp")
    sources = glob.glob(path.join(extensions_dir, "**", "*.cpp"))

    # common code between cuda and rocm platforms, for hipify version [1,0,0] and later.
    source_cuda = glob.glob(path.join(extensions_dir, "**", "*.cu")) + glob.glob(path.join(extensions_dir, "*.cu"))
    sources = [main_source] + sources

    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if HAS_CUDA or FORCE_CUDA:
        extension = CUDAExtension
        sources += source_cuda

        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-O3", # optimization flag
            "-DCUDA_HAS_FP16=1", # 1: should support half-precision
            "-D__CUDA_NO_HALF_OPERATORS__", # disable the default half-precision arithmetic operators provided by CUDA, want to use custom implementations
            "-D__CUDA_NO_HALF_CONVERSIONS__", # disable automatic type conversions involving half-precision fp, avoid unintended errors due to implicit conversions
            "-D__CUDA_NO_HALF2_OPERATORS__", # disable 
        ]

        nvcc_flags_env = os.getenv("NVCC_FLAGS", "")
        
        if nvcc_flags_env != "":
            extra_compile_args["nvcc"].extend(nvcc_flags_env.split(" "))

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            f"{LIBRARY_NAME}._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules

# https://github.com/pypa/sampleproject/blob/db5806e0a3204034c51b1c00dde7d5eb3fa2532e/setup.py 참고
setup(
    name=LIBRARY_NAME, 
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
    install_requires=[
        "torch==2.4.0+cu121",
        "torchvision==0.19.0+cu121",
        "lightning==2.3.3",
        "timm==1.0.8",
    ]
)