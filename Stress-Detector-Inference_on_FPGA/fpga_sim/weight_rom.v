module weight_rom (
    input clk,
    input [3:0] addr,
    output reg signed [7:0] data_out
);

    reg signed [7:0] mem [0:8];

    initial begin
        $readmemh("layer0_features_0_weight.mif", mem);  // Load weights at compile time
    end

    always @(posedge clk) begin
        data_out <= mem[addr];
    end
endmodule
