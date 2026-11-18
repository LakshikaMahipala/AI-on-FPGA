# 🧠 AI on FPGA

**Running trained neural networks as custom Verilog hardware, from PyTorch model to RTL simulation.**

This repository shows the full path from a trained deep learning model to FPGA inference logic: training, INT8 quantization, weight export to FPGA memory files, layer-by-layer RTL design, and verification in simulation.

## Projects

| Project | Summary | Status |
|---|---|---|
| [**Stress Detector — Inference on FPGA**](Stress-Detector-Inference_on_FPGA) | Facial-expression stress classifier (stressed / not stressed) trained in PyTorch on FER2013, quantized to INT8, and implemented as Verilog modules: convolution, ReLU, max-pooling, fully-connected layer and argmax, with weight ROMs initialized from `.mif` files and a UART interface for sending images and receiving predictions | 🟢 RTL complete · verified in ModelSim simulation |

## Pipeline

```
PyTorch training (FER2013)  →  INT8 quantization  →  weight export (.mif)
        →  Verilog layers: conv2d → relu → maxpool → linear → argmax
        →  UART RX/TX top level  →  ModelSim testbench
```

## What's inside `fpga_sim/`

| File | Role |
|---|---|
| `conv2d.v`, `relu.v`, `maxpool.v`, `linear.v`, `argmax.v` | Neural-network layer hardware |
| `weight_rom.v`, `pixel_ram.v`, `*.mif` | On-chip weight and image memories |
| `uart_rx.v`, `uart_tx.v`, `uart_top.v`, `top.v` | Host communication and top-level integration |
| `testbench_uart_top.v`, `simulate.do` | End-to-end simulation testbench and ModelSim script |

## Skills demonstrated
PyTorch · INT8 quantization · Verilog RTL design · fixed-point arithmetic · on-chip memory initialization · UART · ModelSim simulation

## Next steps
- Synthesize for a target FPGA and report resource use (LUT/DSP/BRAM), clock frequency and per-image latency
- Report model accuracy before and after INT8 quantization
- Compare against the same model on a CPU and an NVIDIA GPU

Related work: more FPGA kernels (INT8 FC layers, depthwise Conv1D, binary XNOR-popcount) in [edge-ml-systems](https://github.com/LakshikaMahipala/edge-ml-systems/tree/main/03_fpga).

**Author:** Lakshika Padmamali · [GitHub profile](https://github.com/LakshikaMahipala)
