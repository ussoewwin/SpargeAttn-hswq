# SpargeAttention

Source fork of [thu-ml/SpargeAttn](https://github.com/thu-ml/SpargeAttn). Installation, build prerequisites, CUDA paths, API usage, benchmarks, tests, and citations are documented in the upstream repository; this file only lists pointers.

## Pre-built Windows Wheels (Recommended)

Pre-compiled Windows `.whl` binaries for various PyTorch and CUDA versions (including full 7-generation architecture support for Blackwell/Ampere/Ada/Hopper) are officially published at:
👉 **[GitHub Releases](https://github.com/ussoewwin/SpargeAttn-hswq/releases)**

### Installation Guideline

You do **not** need to build from source (which can take a significant amount of time). Simply locate the wheel that matches your environment (Python, PyTorch, CUDA version) from the releases or wheel repositories and install it directly via pip:

```bash
pip install <matching_wheel_file>.whl
```

## Upstream (authoritative)

- Repository: https://github.com/thu-ml/SpargeAttn  
- README: https://github.com/thu-ml/SpargeAttn/blob/main/README.md  

## Changelog

- Fork-only release history (not the upstream project changelog): [md/CHANGELOG.md](md/CHANGELOG.md)

## License & Attribution

This project is an enhanced fork and derivative work of [SpargeAttn](https://github.com/thu-ml/SpargeAttn), originally developed by the SpargeAttn team (Tsinghua University / thu-ml).

- **Original Work**: Copyright (c) 2025 SpargeAttn team. Licensed under the [Apache License, Version 2.0](LICENSE).
- **Modifications & Enhancements**: Released under the terms of the Apache License, Version 2.0, adhering to all upstream licensing requirements. All original copyright notices, disclaimers, and academic citations are fully preserved.
