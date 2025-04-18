module uart_rx #(
    parameter CLK_FREQ = 50000000,       // 50 MHz
    parameter BAUD_RATE = 115200,
    parameter IMAGE_SIZE = 150528        // 224*224*3
)(
    input clk,
    input rst,
    input rx,  // UART receive line

    output reg [7:0] data_out,
    output reg [16:0] byte_count,       // log2(150528) = 17 bits
    output reg valid,
    output reg done
);

    localparam CLKS_PER_BIT = CLK_FREQ / BAUD_RATE;

    reg [15:0] clk_cnt = 0;
    reg [3:0] bit_idx = 0;
    reg [9:0] rx_shift = 10'b1111111111;
    reg receiving = 0;

    reg [7:0] data_buffer;
    reg [16:0] count;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            clk_cnt <= 0;
            bit_idx <= 0;
            rx_shift <= 10'b1111111111;
            receiving <= 0;
            valid <= 0;
            done <= 0;
            count <= 0;
        end else begin
            valid <= 0;

            // Start bit detection
            if (!receiving && rx == 0) begin
                receiving <= 1;
                clk_cnt <= CLKS_PER_BIT / 2;
                bit_idx <= 0;
            end

            if (receiving) begin
                if (clk_cnt == CLKS_PER_BIT - 1) begin
                    clk_cnt <= 0;
                    rx_shift <= {rx, rx_shift[9:1]};
                    bit_idx <= bit_idx + 1;

                    if (bit_idx == 9) begin
                        receiving <= 0;
                        data_out <= rx_shift[8:1];
                        valid <= 1;

                        if (count < IMAGE_SIZE) begin
                            count <= count + 1;
                            byte_count <= count;
                        end

                        if (count == IMAGE_SIZE - 1) begin
                            done <= 1;  // Inference ready
                        end
                    end
                end else begin
                    clk_cnt <= clk_cnt + 1;
                end
            end
        end
    end
endmodule
