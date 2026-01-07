# CPU-GPU-compute
Simple examples of CPGPU-compute programs in **OpenCL/SYCL**.

It is intended for research and training.

## Requirements:

- C++20 STL implementation

### For OpenCL:

- OpenCL 1.2 or higher;
- [OpenCL-CLHPP](https://github.com/KhronosGroup/OpenCL-CLHPP) header by Khronos

### For SYCL:

- SYCL 2020-compliant compiler (DPC++ or AdaptiveCpp);
- Device Runtime (e.g., Level-Zero for Intel GPU, ROCm for AMD GPU, or OpenCL as a generic backend);

> The SYCL examples were developed and tested only in the **Intel® oneAPI** environment (DPC++ compiler with Level-Zero/CPU runtime).

The original code of the entire project and binary files from the release were debugged and compiled using **ICX/ICPX** compilers in the **Intel® oneAPI** environment.

## Announcement

This repository may be updated over time and/or modified in some way. The order of files in the repository will not be subject to significant changes, but there is no guarantee of this.

## License

```
    Copyright (C) 2026 rabbitGraned
    
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    More detailed: <https://www.gnu.org/licenses/lgpl-3.0.en.html>.
```
