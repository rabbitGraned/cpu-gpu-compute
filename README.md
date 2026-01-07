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

The original code were debugged and tested using **ICX/ICPX** compilers in the **Intel® oneAPI** environment on Windows.

## Announcement

This repository may be updated over time and/or modified in some way. The order of files in the repository will not be subject to significant changes, but there is no guarantee of this.

## Help

To get offline **SPIR-V** of OpenCL kernels, use the **ocloc** utility in oneAPI:

`ocloc.exe compile -device DG2 -file matrix_localmem.cl`

`ocloc.exe disasm -file matrix_localmem_dg2.bin`

For offline compilation via **ocloc**, don't forget to uncomment `#define TILE 16` in _matrix_localmem.cl_ file.

The required name for the "-device" flag can be found in the table below.

### Table of names
| Gen CPU | Gen GPU | Names |
| --- | --- | --- |
| Gen9 | Skylake, Kaby Lake, Coffee Lake | `SKL, KBL, CFL` |
| Gen11 | Ice Lake, Jasper Lake | `ICLLP, EHL` |
| Gen12LP | Tiger Lake, DG1 | `TGLLP, DG1` |
| Gen12HP / Xe-HPG | Alder Lake, Raptor Lake, Arc (Alchemist) | `DG2, MTL, ARL` |
| Xe-HP | Ponte Vecchio | `PVC` |
| Xe2 (Battlemage) | Lunar Lake, Battlemage GPUs | `BMG` (oneAPI 2025.2+) |

On systems where the runtime for SYCL is not installed, you can use the OpenCL backend to run SYCL applications:

`set SYCL_BE=OPENCL && sycl_app.exe`

You can view the description of the SYCL specification [here](https://github.com/KhronosGroup/SYCL-Docs).

## License

CPU-GPU-compute source code is licensed under the [GNU GPL v3](LICENSE).

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
