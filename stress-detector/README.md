# Stress Detector on FPGA

> A CNN that classifies a face image as *stressed* or *not stressed*, running as custom hardware: the image arrives over UART, a one-multiplier datapath runs the network, and the answer goes back over UART.

**Status:** 🟢 RTL verified bit-exact against a Python golden model (CI, Icarus Verilog) · 🟡 training script ready, trained weights not yet exported · 🟡 not yet synthesized on a board

## Network
The VGG-16 from the original notebook has 138 million parameters, far too many for a small FPGA. The hardware runs a network sized for it, with 4,274 parameters:

```
48×48 grayscale ─► conv 3×3, 4 filters ─► ReLU + requant ─► max-pool 2×2 ─► FC 2,116 → 2 ─► argmax
     (uint8)          int8 × int8 → int32     int8 [0,127]      4×23×23 int8      int32 logits    0 / 1
```

| Step | Integer operation (exactly what the hardware does) |
|---|---|
| Input | x = pixel − 128 (in hardware: invert the top bit) |
| Conv | acc = bias + Σ x·w over the 3×3 window |
| ReLU + requant | y = clip(round(acc / 2^SHIFT), 0, 127) |
| Pool | max of each 2×2 block |
| FC + argmax | logit_j = bias_j + Σ feature·w_j; predict the larger logit |

**Worked example:** acc = 5,000 with SHIFT = 8 → 5,000 / 256 = 19.5 → rounds to **20**. acc = −300 → ReLU → **0**.

## Hardware
| Module | Role |
|---|---|
| [`cnn_core.sv`](rtl/cnn_core.sv) | Sequential datapath with one multiplier. Fuses conv, ReLU, requant and pooling, so only the 2,116 pooled features are stored |
| [`uart_rx.sv`](rtl/uart_rx.sv), [`uart_tx.sv`](rtl/uart_tx.sv) | 8N1 UART with an input synchronizer and mid-bit sampling |
| [`stress_top.sv`](rtl/stress_top.sv) | Receives 2,304 pixels → runs the core → sends 1 byte (0 / 1) |

**Latency:** 88,875 clock cycles per image, about 1.8 ms at 50 MHz. Sending the image at 115,200 baud takes about 200 ms, so the UART link, not the compute, is the bottleneck.

## Verification
- `golden/cnn_ref.py`: bit-exact NumPy model, unit-tested against direct loops (`make test`)
- `tb/tb_cnn_core.sv`: 20 images (including all-black, all-white and a gradient); checks the class **and** the exact cycle count
- `tb/tb_stress_top.sv`: end-to-end; sends images over the UART pin and decodes the reply byte
```bash
make test                                       # pytest
make sim                                        # Icarus Verilog, random weights
WEIGHTS=model/exported_weights.npz make sim     # with trained weights
```

## Train and export real weights
```bash
python model/train_and_export.py --data <FER2013>/train --epochs 15
```
This trains the small network (labels as in the notebook: angry/disgust/fear/sad → stressed, happy/neutral → not stressed), quantizes it the way the hardware expects, and reports **float vs INT8-hardware accuracy** on a validation split. _Not yet run; the results go here._

## On the board (next)
- Synthesize `stress_top` (50 MHz clock) and report LUT / DSP / BRAM use
- `python host/send_image.py --port <PORT> face.jpg` sends an image and prints the FPGA's answer

## What was wrong in v0
The first prototype (in git history before November 2026) had all-zero exported weights (floats saved with `%d`), a `.mif` file read with `$readmemh`, one hard-coded 3×3 MAC instead of a network, a 17-bit address for a 150,528-byte image, and a testbench with no checks. v1 fixes each of these: scale-aware quantization, hex weight files generated in Python, a real network sized for the FPGA, parameterised address widths, and self-checking testbenches.

## Layout
```
rtl/        cnn_core.sv · uart_rx.sv · uart_tx.sv · stress_top.sv
tb/         tb_cnn_core.sv · tb_stress_top.sv
golden/     config.py · cnn_ref.py · gen_vectors.py · test_cnn_ref.py
model/      train_and_export.py (PyTorch: train, quantize, export)
host/       send_image.py (PC → FPGA over serial)
notebooks/  vgg16_software_baseline.ipynb (original VGG-16 software model)
docs/       v0_tutorial.md
```
