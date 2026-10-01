"""
Copyright (c) 2025 by SpargeAttn team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import sys
from pathlib import Path
import subprocess
from packaging.version import parse, Version
from typing import List, Set
import warnings

from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

HAS_SM90 = False
SAGE2PP_ENABLED = True

# Windows CreateProcess 32k command-line limit fix for link.exe
if os.name == "nt":

    def _create_safe_spawn(orig_spawn):
        def _safe_spawn(self, cmd, *args, **kwargs):
            if len(cmd) > 1 and ("link.exe" in cmd[0].lower() or cmd[0].lower().endswith("link")):
                total_len = sum(len(c) for c in cmd)
                if total_len > 8000:
                    import tempfile
                    with tempfile.NamedTemporaryFile("w", suffix=".rsp", delete=False, encoding="utf-8") as f:
                        for arg in cmd[1:]:
                            if " " in arg and not (arg.startswith('"') and arg.endswith('"')):
                                f.write(f'"{arg}"\n')
                            else:
                                f.write(f'{arg}\n')
                        rsp_name = f.name
                    try:
                        return orig_spawn(self, [cmd[0], f"@{rsp_name}"], *args, **kwargs)
                    finally:
                        try:
                            os.remove(rsp_name)
                        except Exception:
                            pass
            return orig_spawn(self, cmd, *args, **kwargs)
        return _safe_spawn

    for mod_name in [
        "setuptools._distutils.compilers.C.msvc",
        "setuptools._distutils._msvccompiler",
        "distutils._msvccompiler",
    ]:
        try:
            mod = __import__(mod_name, fromlist=["Compiler", "MSVCCompiler"])
            cls = getattr(mod, "Compiler", getattr(mod, "MSVCCompiler", None))
            if cls and hasattr(cls, "spawn"):
                cls.spawn = _create_safe_spawn(cls.spawn)
        except Exception:
            pass



def run_instantiations(src_dir: str):
    base_path = Path(src_dir)
    existing_cu = list(base_path.glob("*.cu"))
    if existing_cu:
        return
    py_files = [
        path for path in base_path.rglob('*.py')
        if path.is_file()
    ]

    for py_file in py_files:
        print(f"Running: {py_file}")
        subprocess.check_call([sys.executable, str(py_file)], cwd=str(base_path))

def get_instantiations(src_dir: str):
    # get all .cu files under src_dir
    base_path = Path(src_dir)
    return [
        os.path.join(src_dir, str(path.relative_to(base_path)))
        for path in base_path.rglob('*')
        if path.is_file() and path.suffix == ".cu"
    ]

# Supported NVIDIA GPU architectures.
SUPPORTED_ARCHS = {"8.0", "8.6", "8.7", "8.9", "9.0", "10.0", "12.0", "12.1"}

# Compiler flags.
# Windows/MSVC note (HSWQ fork): -fopenmp/-lgomp are gcc-only; CUDA 13.x CCCL
# requires the standard-conforming preprocessor (/Zc:preprocessor) and the
# assert include. Mirror the flag set proven in the SageAttention Windows fork.
import os as _os
if _os.name == "nt":
    CXX_FLAGS = ["/O2", "/std:c++20", "/Zc:preprocessor", "/Zc:__cplusplus", "-DENABLE_BF16"]
    NVCC_FLAGS = [
        "-O3",
        "-std=c++20",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "--use_fast_math",
        "--threads=8",
        "-Xptxas=-v",
        "-diag-suppress=174", # suppress the specific warning
        "-diag-suppress=177",
        "-diag-suppress=221",
        "-D_WIN32=1",
        "-Xcompiler", "/Zc:preprocessor",   # CUDA 13 CCCL requires standard-conforming pp
        "-Xcompiler", "/std:c++20",
        "-Xcompiler", "/Zc:__cplusplus",
    ]
else:
    CXX_FLAGS = ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17", "-DENABLE_BF16"]
    NVCC_FLAGS = [
        "-O3",
        "-std=c++17",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "--use_fast_math",
        "--threads=8",
        "-Xptxas=-v",
        "-diag-suppress=174", # suppress the specific warning
        "-Xcompiler", "-include,cassert", # fix error occurs when compiling for SM90+ with newer CUDA toolkits
    ]

if _os.name != "nt":
    ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
    CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
    NVCC_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]

if CUDA_HOME is None:
    raise RuntimeError(
        "Cannot find CUDA_HOME. CUDA must be available to build the package.")

def get_nvcc_cuda_version(cuda_dir: str) -> Version:
    """Get the CUDA version from nvcc.

    Adapted from https://github.com/NVIDIA/apex/blob/8b7a1ff183741dd8f9b87e7bafd04cfde99cea28/setup.py
    """
    nvcc_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"],
                                          universal_newlines=True)
    output = nvcc_output.split()
    release_idx = output.index("release") + 1
    nvcc_cuda_version = parse(output[release_idx].split(",")[0])
    return nvcc_cuda_version

def get_torch_arch_list() -> Set[str]:
    # TORCH_CUDA_ARCH_LIST can have one or more architectures,
    # e.g. "8.0" or "7.5,8.0,8.6+PTX". Here, the "8.6+PTX" option asks the
    # compiler to additionally include PTX code that can be runtime-compiled
    # and executed on the 8.6 or newer architectures. While the PTX code will
    # not give the best performance on the newer architectures, it provides
    # forward compatibility.
    env_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
    if env_arch_list is None:
        return set()

    raw_arch_list = env_arch_list.replace(" ", ";").split(";")
    torch_arch_list = set()
    for item in raw_arch_list:
        item = item.strip()
        if not item:
            continue
        ptx = "+PTX" if item.endswith("+PTX") else ""
        base = item[:-4] if ptx else item
        if base in ("80", "86", "87", "89", "90"):
            base = f"{base[0]}.{base[1]}"
        elif base in ("100", "120", "121"):
            base = f"{base[:2]}.{base[2:]}"
        torch_arch_list.add(base + ptx)

    if not torch_arch_list:
        return set()

    # Filter out the invalid architectures and print a warning.
    valid_archs = SUPPORTED_ARCHS.union({s + "+PTX" for s in SUPPORTED_ARCHS})
    arch_list = torch_arch_list.intersection(valid_archs)
    # If none of the specified architectures are valid, raise an error.
    if not arch_list:
        raise RuntimeError(
            "None of the CUDA architectures in `TORCH_CUDA_ARCH_LIST` env "
            f"variable ({env_arch_list}) is supported. "
            f"Supported CUDA architectures are: {valid_archs}.")
    invalid_arch_list = torch_arch_list - valid_archs
    if invalid_arch_list:
        warnings.warn(
            f"Unsupported CUDA architectures ({invalid_arch_list}) are "
            "excluded from the `TORCH_CUDA_ARCH_LIST` env variable "
            f"({env_arch_list}). Supported CUDA architectures are: "
            f"{valid_archs}.")
    return arch_list

# First, check the TORCH_CUDA_ARCH_LIST environment variable.
compute_capabilities = get_torch_arch_list()
if not compute_capabilities:
    # If TORCH_CUDA_ARCH_LIST is not defined or empty, target all available
    # GPUs on the current machine.
    device_count = torch.cuda.device_count()
    for i in range(device_count):
        major, minor = torch.cuda.get_device_capability(i)
        if major < 8:
            raise RuntimeError(
                "GPUs with compute capability below 8.0 are not supported.")
        compute_capabilities.add(f"{major}.{minor}")

nvcc_cuda_version = get_nvcc_cuda_version(CUDA_HOME)
if not compute_capabilities:
    raise RuntimeError("No GPUs found. Please specify the target GPU architectures or build on a machine with GPUs.")

# Validate the NVCC CUDA version.
if nvcc_cuda_version < Version("12.0"):
    raise RuntimeError("CUDA 12.0 or higher is required to build the package.")
if nvcc_cuda_version < Version("12.4"):
    if any(cc.startswith("8.9") for cc in compute_capabilities):
        raise RuntimeError(
            "CUDA 12.4 or higher is required for compute capability 8.9.")
    if any(cc.startswith("9.0") for cc in compute_capabilities):
        raise RuntimeError(
            "CUDA 12.4 or higher is required for compute capability 9.0.")
if nvcc_cuda_version < Version("12.8"):
    warnings.warn("CUDA 12.8 or higher is required for Sage2++")
    SAGE2PP_ENABLED = False

# Add target compute capabilities to NVCC flags.
for capability in compute_capabilities:
    num = capability.replace(".", "")
    if num == '90' and not any(cc.startswith("10.") or cc.startswith("12.") for cc in compute_capabilities):
        num = '90a'
        HAS_SM90 = True
        CXX_FLAGS += ["-DHAS_SM90"]
    elif num == '90':
        num = '90a'
    if num == '80' or num == '86' or num == '87':
        SAGE2PP_ENABLED = False
    
    NVCC_FLAGS += ["-gencode", f"arch=compute_{num},code=sm_{num}"]
    if capability.endswith("+PTX"):
        NVCC_FLAGS += ["-gencode", f"arch=compute_{num},code=compute_{num}"]

if SAGE2PP_ENABLED:
    CXX_FLAGS += ["-DSAGE2PP_ENABLED"]

ext_modules = []

# run_instantiations("csrc/qattn/instantiations_sm80")
# run_instantiations("csrc/qattn/instantiations_sm89")
# run_instantiations("csrc/qattn/instantiations_sm90")

sources = [
    "csrc/qattn/pybind.cpp",
    "csrc/qattn/qk_int_sv_f16_cuda_sm80.cu",
    "csrc/qattn/qk_int_sv_f8_cuda_sm89.cu",
] + get_instantiations("csrc/qattn/instantiations_sm80") + get_instantiations("csrc/qattn/instantiations_sm89")

if HAS_SM90:
    sources += ["csrc/qattn/qk_int_sv_f8_cuda_sm90.cu", ]
    sources += get_instantiations("csrc/qattn/instantiations_sm90")

qattn_extension = CUDAExtension(
    name="spas_sage_hswq_attn._qattn",
    sources=sources,
    extra_compile_args={
        "cxx": CXX_FLAGS,
        "nvcc": NVCC_FLAGS,
    },
    extra_link_args=['-lcuda'],
)
ext_modules.append(qattn_extension)

fused_extension = CUDAExtension(
    name="spas_sage_hswq_attn._fused",
    sources=["csrc/fused/pybind.cpp", "csrc/fused/fused.cu"],
    extra_compile_args={
        "cxx": CXX_FLAGS,
        "nvcc": NVCC_FLAGS,
    },
)
ext_modules.append(fused_extension)

def get_package_version():
    base_version = "1.0.0"
    torch_version_raw = parse(torch.__version__)
    torch_version = f"{torch_version_raw.major}.{torch_version_raw.minor}.{torch_version_raw.micro}" if hasattr(torch_version_raw, 'micro') else f"{torch_version_raw.major}.{torch_version_raw.minor}"
    
    cuda_version = "132"
    if torch.version.cuda:
        cuda_version = torch.version.cuda.replace(".", "")
    
    cxx11_abi = "TRUE"
    if hasattr(torch._C, "_GLIBCXX_USE_CXX11_ABI"):
        cxx11_abi = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()
        
    local_version = f"cu{cuda_version}torch{torch_version}cxx11abi{cxx11_abi}"
    return f"{base_version}+{local_version}"

setup(
    name='spas_sage_hswq_attn', 
    version=get_package_version(),  
    author='Jintao Zhang, Chendong Xiang, Haofeng Huang',  
    author_email='jt-zhang6@gmail.com', 
    # Ship ONLY the attention package. Generic top-level names such as 'tools',
    # 'evaluate' and 'inference_examples' must never be installed: they shadow
    # same-named packages other consumers rely on.
    packages=["spas_sage_hswq_attn"],  
    description='Accurate and efficient Sparse SageAttention.',  
    long_description=open('README.md', encoding='utf-8').read(),  
    long_description_content_type='text/markdown', 
    url='https://github.com/thu-ml/SpargeAttn', 
    license='BSD 3-Clause License', 
    python_requires='>=3.9', 
    classifiers=[  
        'Development Status :: 3 - Alpha', 
        'Intended Audience :: Developers',  
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3', 
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
