// UART transmitter, 8N1. Pulse `send` with `data` while `busy` is low.
module uart_tx #(
    parameter int CLKS_PER_BIT = 434
) (
    input  logic       clk,
    input  logic       rst,
    input  logic       send,
    input  logic [7:0] data,
    output logic       tx,        // serial output, idle high
    output logic       busy
);
    logic [9:0] frame;            // {stop, data[7:0], start}
    int clk_cnt, bit_idx;

    always_ff @(posedge clk) begin
        if (rst) begin
            tx <= 1'b1; busy <= 1'b0; clk_cnt <= 0; bit_idx <= 0;
        end else if (!busy) begin
            tx <= 1'b1;
            if (send) begin
                frame <= {1'b1, data, 1'b0};
                busy <= 1'b1; clk_cnt <= 0; bit_idx <= 0;
                tx <= 1'b0;                        // start bit
            end
        end else if (clk_cnt == CLKS_PER_BIT - 1) begin
            clk_cnt <= 0;
            if (bit_idx == 9) begin
                busy <= 1'b0; tx <= 1'b1;
            end else begin
                bit_idx <= bit_idx + 1;
                tx <= frame[bit_idx + 1];
            end
        end else clk_cnt <= clk_cnt + 1;
    end
endmodule
