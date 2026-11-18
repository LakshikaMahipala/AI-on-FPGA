// Runs every test image through cnn_core and checks the class and cycle count
// against golden/gen_vectors.py.
`include "config.svh"
`timescale 1ns/1ps
module tb_cnn_core;
    localparam int IMG = `IMG, N_PIX = `IMG * `IMG, N = `N_IMAGES;

    logic clk = 0, rst = 1, img_we = 0, start = 0, busy, done;
    logic [$clog2(N_PIX)-1:0] img_addr;
    logic [7:0] img_data, class_out;
    logic [7:0] images[N*N_PIX];
    logic [7:0] expected[N];
    int errors = 0, cycles;
    time t_start;

    always #5 clk = ~clk;

    cnn_core #(.IMG(IMG), .K(`K), .C_OUT(`C_OUT), .N_CLASSES(`N_CLASSES), .SHIFT(`SHIFT)) dut (
        .clk(clk), .rst(rst), .img_we(img_we), .img_addr(img_addr), .img_data(img_data),
        .start(start), .busy(busy), .done(done), .class_out(class_out));

    initial begin
        $readmemh("vectors/images.hex", images);
        $readmemh("vectors/expected_class.hex", expected);
        repeat (2) @(posedge clk);
        rst <= 0;
        for (int t = 0; t < N; t++) begin
            for (int p = 0; p < N_PIX; p++) begin
                @(posedge clk);
                img_we <= 1; img_addr <= p; img_data <= images[t*N_PIX + p];
            end
            @(posedge clk) begin img_we <= 0; start <= 1; end
            @(posedge clk) start <= 0;          // start is sampled on this edge
            t_start = $time;
            @(posedge done);
            cycles = ($time - t_start) / 10;
            #1;
            if (class_out !== expected[t] || cycles != `EXPECTED_CYCLES) begin
                $display("MISMATCH image %0d: class %0d (expected %0d), %0d cycles (expected %0d)",
                         t, class_out, expected[t], cycles, `EXPECTED_CYCLES);
                errors++;
            end
        end
        if (errors == 0)
            $display("PASS tb_cnn_core: %0d images, %0d cycles each", N, `EXPECTED_CYCLES);
        else $display("FAIL tb_cnn_core: %0d errors", errors);
        $finish;
    end
endmodule
