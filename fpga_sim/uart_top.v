module uart_top (
    input clk,
    input rst,
    input uart_rx_pin,    // From PC/webapp
    output uart_tx_pin    // Back to PC/webapp
);

    // UART RX to RAM
    wire [7:0] pixel_byte;
    wire [16:0] pixel_addr;
    wire valid_rx, image_done;

    uart_rx rx_inst (
        .clk(clk),
        .rst(rst),
        .rx(uart_rx_pin),
        .data_out(pixel_byte),
        .byte_count(pixel_addr),
        .valid(valid_rx),
        .done(image_done)
    );

    // RAM to Inference Core
    wire [7:0] ram_out;
    reg [16:0] read_addr = 0;
    wire [0:0] stress_result;
    wire prediction_valid;

    pixel_ram ram_inst (
        .clk(clk),
        .write_en(valid_rx),
        .write_addr(pixel_addr),
        .write_data(pixel_byte),
        .read_addr(read_addr),
        .read_data(ram_out)
    );

    // ROM address tracker for 3x3 conv weights
    reg [3:0] weight_addr = 0;
    always @(posedge clk or posedge rst) begin
        if (rst)
            weight_addr <= 0;
        else if (image_done && weight_addr < 9)
            weight_addr <= weight_addr + 1;
    end

    // Connect weight ROM
    wire signed [7:0] weight_from_rom;

    weight_rom conv_weight_rom (
        .clk(clk),
        .addr(weight_addr),
        .data_out(weight_from_rom)
    );

    // Read RAM when image is fully received
    always @(posedge clk or posedge rst) begin
        if (rst)
            read_addr <= 0;
        else if (image_done && read_addr < 150528)
            read_addr <= read_addr + 1;
    end

    // Inference Core
    top inference_core (
        .clk(clk),
        .rst(rst),
        .pixel_in(ram_out),
        .weight_in(weight_from_rom),          // ✅ real weight stream
        .valid_pixel(image_done),
        .stress_prediction(stress_result),
        .prediction_valid(prediction_valid)
    );

    // UART TX
    uart_tx tx_inst (
        .clk(clk),
        .rst(rst),
        .data_in({7'd0, stress_result}),      // Pad 1-bit result to 8-bit byte
        .start(prediction_valid),
        .tx(uart_tx_pin),
        .busy()
    );
endmodule
