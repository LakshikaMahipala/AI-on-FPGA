module linear #(
    parameter INPUT_SIZE = 4096,
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH = 32
)(
    input clk,
    input rst,

    input valid_in,
    input signed [DATA_WIDTH-1:0] x_in,
    input signed [DATA_WIDTH-1:0] w_in,

    output reg signed [ACC_WIDTH-1:0] y_out,
    output reg valid_out
);

    reg signed [ACC_WIDTH-1:0] acc;
    reg [$clog2(INPUT_SIZE):0] count;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            acc <= 0;
            count <= 0;
            y_out <= 0;
            valid_out <= 0;
        end else if (valid_in) begin
            acc <= acc + (x_in * w_in);
            count <= count + 1;

            if (count == INPUT_SIZE - 1) begin
                y_out <= acc + (x_in * w_in);
                valid_out <= 1;
                acc <= 0;
                count <= 0;
            end else begin
                valid_out <= 0;
            end
        end else begin
            valid_out <= 0;
        end
    end
endmodule
