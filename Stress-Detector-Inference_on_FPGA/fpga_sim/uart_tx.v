module uart_tx #(
    parameter CLK_FREQ = 50000000,       // 50 MHz
    parameter BAUD_RATE = 115200
)(
    input clk,
    input rst,

    input [7:0] data_in,
    input start,  // set high for 1 cycle to send

    output reg tx,        // UART TX line
    output reg busy       // High while transmitting
);

    localparam CLKS_PER_BIT = CLK_FREQ / BAUD_RATE;

    reg [3:0] bit_idx = 0;
    reg [15:0] clk_cnt = 0;
    reg [9:0] tx_shift = 10'b1111111111;

    reg sending = 0;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            tx <= 1'b1;
            clk_cnt <= 0;
            bit_idx <= 0;
            busy <= 0;
            sending <= 0;
        end else begin
            if (start && !sending) begin
                // Load start + data + stop bits
                tx_shift <= {1'b1, data_in, 1'b0};  // LSB first: start bit = 0
                sending <= 1;
                busy <= 1;
                bit_idx <= 0;
                clk_cnt <= 0;
            end

            if (sending) begin
                if (clk_cnt == CLKS_PER_BIT - 1) begin
                    clk_cnt <= 0;
                    tx <= tx_shift[bit_idx];
                    bit_idx <= bit_idx + 1;

                    if (bit_idx == 9) begin
                        sending <= 0;
                        busy <= 0;
                        tx <= 1;  // Idle line
                    end
                end else begin
                    clk_cnt <= clk_cnt + 1;
                end
            end
        end
    end
endmodule
