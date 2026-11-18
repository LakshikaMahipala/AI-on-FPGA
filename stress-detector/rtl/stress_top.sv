// FPGA top level: receive a 48x48 grayscale image over UART, classify it,
// send back one byte: 0x00 = not stressed, 0x01 = stressed.
//
// Host protocol: send exactly IMG*IMG pixel bytes (row-major); the result byte
// follows. Then the next image can be sent. (See host/send_image.py.)
module stress_top #(
    parameter int CLKS_PER_BIT = 434,   // 50 MHz / 115200 baud
    parameter int IMG   = 48,
    parameter int SHIFT = 8
) (
    input  logic clk,
    input  logic rst,
    input  logic uart_rx_pin,
    output logic uart_tx_pin,
    output logic busy_led
);
    localparam int N_PIX = IMG * IMG;

    logic [7:0] rx_byte, class_out;
    logic       rx_valid, core_busy, core_done, tx_busy, start;
    logic [$clog2(N_PIX)-1:0] pix_addr;

    uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_rx (
        .clk(clk), .rst(rst), .rx(uart_rx_pin), .data(rx_byte), .valid(rx_valid));

    // Pixel counter: the last pixel of an image starts inference.
    always_ff @(posedge clk) begin
        if (rst) begin
            pix_addr <= '0; start <= 1'b0;
        end else begin
            start <= 1'b0;
            if (rx_valid && !core_busy) begin
                if (pix_addr == N_PIX - 1) begin pix_addr <= '0; start <= 1'b1; end
                else pix_addr <= pix_addr + 1'b1;
            end
        end
    end

    cnn_core #(.IMG(IMG), .SHIFT(SHIFT)) u_core (
        .clk(clk), .rst(rst),
        .img_we(rx_valid && !core_busy), .img_addr(pix_addr), .img_data(rx_byte),
        .start(start), .busy(core_busy), .done(core_done), .class_out(class_out));

    uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_tx (
        .clk(clk), .rst(rst), .send(core_done), .data(class_out),
        .tx(uart_tx_pin), .busy(tx_busy));

    assign busy_led = core_busy | tx_busy;
endmodule
