# FPGA AI Accelerator

> Design of a neural-network inference accelerator on FPGA.

**Status:** 🚧 In progress. Code is being added.

## Goal
_To be completed with the design: target network(s), target FPGA board, and the performance goal (latency, throughput, resources)._

## Architecture
_Block diagram and description: compute array, on-chip buffers, control, host interface._

## Results
| Metric | Value |
|---|---|
| Target device | – |
| Clock frequency | – |
| LUT / DSP / BRAM | – |
| Latency per inference | – |
| Accuracy vs software model | – |

## Layout
```
rtl/       synthesizable Verilog / SystemVerilog
tb/        self-checking testbenches (compare against golden vectors)
golden/    bit-exact Python reference model and test-vector generator
scripts/   simulation / synthesis scripts (Makefile, Vivado or Quartus Tcl)
docs/      architecture notes and diagrams
```

## Adding code (conventions)
- Every RTL module gets a testbench in `tb/` that prints `PASS` or `FAIL`.
- Expected outputs come from the Python model in `golden/`, never hand-typed numbers.
- Weights and test vectors are generated as plain hex files for `$readmemh`.
- Record results (simulation, synthesis reports) in the table above, with the tool version.
