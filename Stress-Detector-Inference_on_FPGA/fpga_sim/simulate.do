vlib work
vlog *.v
vsim work.testbench_uart_top

add wave -divider "Clocks & Reset"
add wave clk
add wave rst

add wave -divider "UART"
add wave uart_rx_pin
add wave uart_tx_pin

add wave -divider "RAM"
add wave dut.pixel_addr
add wave dut.ram_out
add wave dut.read_addr

add wave -divider "ROM"
add wave dut.weight_addr
add wave dut.weight_from_rom

add wave -divider "Inference"
add wave dut.stress_result
add wave dut.prediction_valid

run 2 ms
