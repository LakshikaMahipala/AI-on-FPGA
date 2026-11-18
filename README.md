# AI on FPGA

**Neural-network inference in custom FPGA hardware, from trained model to Verilog.**

[![CI](https://github.com/LakshikaMahipala/AI-on-FPGA/actions/workflows/ci.yml/badge.svg)](https://github.com/LakshikaMahipala/AI-on-FPGA/actions/workflows/ci.yml)

| Project | Summary | Status |
|---|---|---|
| [**FPGA AI Accelerator**](fpga-ai-accelerator) | Design of a neural-network inference accelerator | 🚧 In progress |
| [**Stress Detector**](stress-detector) | Face image in over UART → CNN in custom hardware → stressed / not stressed. Bit-exact against a Python golden model | 🟢 RTL verified in CI · 🟡 board + trained weights pending |

Related, verified FPGA work (INT8 FC and depthwise-conv kernels, binary XNOR-popcount core, all simulated in CI): [edge-ml-systems/projects](https://github.com/LakshikaMahipala/edge-ml-systems/tree/main/projects).

**Author:** Lakshika Padmamali · [GitHub profile](https://github.com/LakshikaMahipala)
