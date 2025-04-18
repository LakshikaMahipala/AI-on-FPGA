module top #(
    parameter PIXEL_WIDTH = 8,
    parameter ACC_WIDTH = 32
)(
    input clk,
    input rst,

    input signed [PIXEL_WIDTH-1:0] pixel_in,    // streamed input pixel
    input signed [PIXEL_WIDTH-1:0] weight_in,   // streamed weight
    input valid_pixel,                          // one pixel/weight per clock

    output reg [0:0] stress_prediction,         // final class output
    output reg prediction_valid                 // high when prediction is ready
);

    // Intermediate signals
    wire signed [ACC_WIDTH-1:0] conv_out;
    wire valid_conv;
    wire signed [ACC_WIDTH-1:0] relu_out;
    wire signed [ACC_WIDTH-1:0] logit_0, logit_1;
    wire valid0, valid1;
    wire [0:0] predicted_class;

    // Convolution layer (3×3)
    conv2d #(.DATA_WIDTH(PIXEL_WIDTH), .ACC_WIDTH(ACC_WIDTH)) conv_layer (
        .clk(clk),
        .rst(rst),
        .pixel_in(pixel_in),
        .weight_in(weight_in),
        .valid_in(valid_pixel),
        .acc_out(conv_out),
        .valid_out(valid_conv)
    );

    // ReLU activation
    relu #(.DATA_WIDTH(ACC_WIDTH)) relu_block (
        .in_val(conv_out),
        .out_val(relu_out)
    );

    // Fully Connected Layer 0 (Class 0)
    linear #(.INPUT_SIZE(1), .DATA_WIDTH(PIXEL_WIDTH), .ACC_WIDTH(ACC_WIDTH)) fc0 (
        .clk(clk),
        .rst(rst),
        .valid_in(valid_conv),
        .x_in(relu_out[7:0]),   // quantized output to INT8
        .w_in(8'sd1),           // replace with ROM preload or streaming weight
        .y_out(logit_0),
        .valid_out(valid0)
    );

    // Fully Connected Layer 1 (Class 1)
    linear #(.INPUT_SIZE(1), .DATA_WIDTH(PIXEL_WIDTH), .ACC_WIDTH(ACC_WIDTH)) fc1 (
        .clk(clk),
        .rst(rst),
        .valid_in(valid_conv),
        .x_in(relu_out[7:0]),
        .w_in(8'sd2),           // replace with ROM preload or streaming weight
        .y_out(logit_1),
        .valid_out(valid1)
    );

    // ArgMax decision block
    argmax #(.DATA_WIDTH(ACC_WIDTH)) argmax_unit (
        .in0(logit_0),
        .in1(logit_1),
        .predicted_class(predicted_class)
    );

    // Output register logic
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            stress_prediction <= 0;
            prediction_valid <= 0;
        end else if (valid0 && valid1) begin
            stress_prediction <= predicted_class;
            prediction_valid <= 1;
        end else begin
            prediction_valid <= 0;
        end
    end
endmodule
