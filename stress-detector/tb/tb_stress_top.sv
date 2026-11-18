// End-to-end test: send images over the UART pin, decode the reply byte,
// compare with the golden prediction. A fast baud rate keeps simulation short.
`include "config.svh"
`timescale 1ns/1ps
module tb_stress_top;
    localparam int CPB = 8;                           // clocks per UART bit (simulation only)
    localparam int N_PIX = `IMG * `IMG;
    localparam int N_TEST = 3;                        // UART is slow: test the first images

    logic clk = 0, rst = 1, rx = 1, tx, busy_led;
    logic [7:0] images[`N_IMAGES*N_PIX];
    logic [7:0] expected[`N_IMAGES];
    logic [7:0] reply;
    int errors = 0;

    always #5 clk = ~clk;

    stress_top #(.CLKS_PER_BIT(CPB), .IMG(`IMG), .SHIFT(`SHIFT)) dut (
        .clk(clk), .rst(rst), .uart_rx_pin(rx), .uart_tx_pin(tx), .busy_led(busy_led));

    task automatic send_byte(input logic [7:0] b);
        rx = 0; repeat (CPB) @(posedge clk);                       // start bit
        for (int k = 0; k < 8; k++) begin rx = b[k]; repeat (CPB) @(posedge clk); end
        rx = 1; repeat (CPB) @(posedge clk);                       // stop bit
    endtask

    task automatic receive_byte(output logic [7:0] b);
        @(negedge tx);                                             // start bit begins
        repeat (CPB + CPB/2) @(posedge clk);                       // middle of data bit 0
        for (int k = 0; k < 8; k++) begin b[k] = tx; repeat (CPB) @(posedge clk); end
    endtask

    initial begin
        $readmemh("vectors/images.hex", images);
        $readmemh("vectors/expected_class.hex", expected);
        repeat (4) @(posedge clk);
        rst = 0;
        for (int t = 0; t < N_TEST; t++) begin
            fork
                for (int p = 0; p < N_PIX; p++) send_byte(images[t*N_PIX + p]);
                receive_byte(reply);
            join
            if (reply !== expected[t]) begin
                $display("MISMATCH image %0d: FPGA replied %0d, expected %0d", t, reply, expected[t]);
                errors++;
            end
            repeat (2*CPB) @(posedge clk);
        end
        if (errors == 0) $display("PASS tb_stress_top: %0d images classified over UART", N_TEST);
        else $display("FAIL tb_stress_top: %0d errors", errors);
        $finish;
    end
endmodule
