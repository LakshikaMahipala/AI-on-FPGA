module conv2d #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH = 32
)(
    input clk,
    input rst,

    input signed [DATA_WIDTH-1:0] pixel_in,
    input signed [DATA_WIDTH-1:0] weight_in,
    input valid_in,

    output reg signed [ACC_WIDTH-1:0] acc_out,
    output reg valid_out
);

    reg signed [ACC_WIDTH-1:0] acc_reg;
    reg [3:0] counter;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            acc_reg <= 0;
            acc_out <= 0;
            valid_out <= 0;
            counter <= 0;
        end else if (valid_in) begin
            acc_reg <= acc_reg + (pixel_in * weight_in);
            counter <= counter + 1;

            // After 9 pixels for 3x3 conv
            if (counter == 8) begin
                acc_out <= acc_reg + (pixel_in * weight_in);
                acc_reg <= 0;
                counter <= 0;
                valid_out <= 1;
            end else begin
                valid_out <= 0;
            end
        end else begin
            valid_out <= 0;
        end
    end
endmodule
