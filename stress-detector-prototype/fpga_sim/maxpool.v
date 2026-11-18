module maxpool #(
    parameter DATA_WIDTH = 32
)(
    input signed [DATA_WIDTH-1:0] a,
    input signed [DATA_WIDTH-1:0] b,
    input signed [DATA_WIDTH-1:0] c,
    input signed [DATA_WIDTH-1:0] d,
    output signed [DATA_WIDTH-1:0] max_out
);

    wire signed [DATA_WIDTH-1:0] max1;
    wire signed [DATA_WIDTH-1:0] max2;

    assign max1 = (a > b) ? a : b;
    assign max2 = (c > d) ? c : d;
    assign max_out = (max1 > max2) ? max1 : max2;

endmodule
