module relu #(
    parameter DATA_WIDTH = 32
)(
    input signed [DATA_WIDTH-1:0] in_val,
    output signed [DATA_WIDTH-1:0] out_val
);

    assign out_val = (in_val < 0) ? 0 : in_val;

endmodule
