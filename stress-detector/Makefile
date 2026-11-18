# make test  -> golden-model unit tests (pytest)
# make sim   -> generate vectors, simulate core + full UART system (Icarus Verilog >= 11)
# WEIGHTS=model/exported_weights.npz make sim   -> verify with trained weights
PYTHON ?= python3
IVERILOG = iverilog -g2012 -I vectors
WEIGHTS ?=
.PHONY: all test vectors sim clean
all: test sim

test:
	cd golden && $(PYTHON) -m pytest -q

vectors:
	cd golden && $(PYTHON) gen_vectors.py --out ../vectors $(if $(WEIGHTS),--weights ../$(WEIGHTS))

sim: vectors
	@mkdir -p build
	$(IVERILOG) -o build/tb_cnn_core.vvp rtl/cnn_core.sv tb/tb_cnn_core.sv
	vvp -n build/tb_cnn_core.vvp | tee build/tb_cnn_core.log
	@grep -q "^PASS" build/tb_cnn_core.log
	$(IVERILOG) -o build/tb_stress_top.vvp rtl/uart_rx.sv rtl/uart_tx.sv rtl/cnn_core.sv rtl/stress_top.sv tb/tb_stress_top.sv
	vvp -n build/tb_stress_top.vvp | tee build/tb_stress_top.log
	@grep -q "^PASS" build/tb_stress_top.log
	@echo "All stress-detector testbenches passed."

clean:
	rm -rf build vectors
