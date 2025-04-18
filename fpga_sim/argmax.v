module argmax #(
    parameter DATA_WIDTH = 32
)(
    input signed [DATA_WIDTH-1:0] in0,
    input signed [DATA_WIDTH-1:0] in1,
    output reg [0:0] predicted_class
);

    always @(*) begin
        if (in1 > in0)
            predicted_class = 1;
        else
            predicted_class = 0;
    end

endmodule
