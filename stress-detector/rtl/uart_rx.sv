// UART receiver, 8 data bits, no parity, 1 stop bit (8N1).
// CLKS_PER_BIT = clock frequency / baud rate (e.g. 50 MHz / 115200 = 434).
module uart_rx #(
    parameter int CLKS_PER_BIT = 434
) (
    input  logic       clk,
    input  logic       rst,
    input  logic       rx,        // serial input, idle high
    output logic [7:0] data,
    output logic       valid      // one-cycle pulse when `data` holds a new byte
);
    typedef enum logic [1:0] {IDLE, START, DATA, STOP} state_t;
    state_t state;
    int clk_cnt, bit_idx;
    logic rx_sync1, rx_sync;      // two-flop synchronizer for the asynchronous input

    always_ff @(posedge clk) begin
        rx_sync1 <= rx;
        rx_sync  <= rx_sync1;
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE; valid <= 1'b0; clk_cnt <= 0; bit_idx <= 0;
        end else begin
            valid <= 1'b0;
            case (state)
                IDLE: if (!rx_sync) begin state <= START; clk_cnt <= 0; end
                START: begin                       // check the start bit in its middle
                    if (clk_cnt == CLKS_PER_BIT/2 - 1) begin
                        clk_cnt <= 0;
                        if (!rx_sync) begin state <= DATA; bit_idx <= 0; end
                        else state <= IDLE;        // glitch, not a start bit
                    end else clk_cnt <= clk_cnt + 1;
                end
                DATA: begin                        // sample each bit in its middle, LSB first
                    if (clk_cnt == CLKS_PER_BIT - 1) begin
                        clk_cnt <= 0;
                        data[bit_idx] <= rx_sync;
                        if (bit_idx == 7) state <= STOP;
                        else bit_idx <= bit_idx + 1;
                    end else clk_cnt <= clk_cnt + 1;
                end
                STOP: begin
                    if (clk_cnt == CLKS_PER_BIT - 1) begin
                        clk_cnt <= 0;
                        state   <= IDLE;
                        valid   <= rx_sync;        // drop the byte if the stop bit is missing
                    end else clk_cnt <= clk_cnt + 1;
                end
            endcase
        end
    end
endmodule
