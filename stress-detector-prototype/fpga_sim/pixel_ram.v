module pixel_ram #(
    parameter ADDR_WIDTH = 17,
    parameter DATA_WIDTH = 8,
    parameter RAM_DEPTH = 150528
)(
    input clk,

    // Write interface
    input write_en,
    input [ADDR_WIDTH-1:0] write_addr,
    input [DATA_WIDTH-1:0] write_data,

    // Read interface
    input [ADDR_WIDTH-1:0] read_addr,
    output reg [DATA_WIDTH-1:0] read_data
);

    reg [DATA_WIDTH-1:0] mem [0:RAM_DEPTH-1];

    // Write
    always @(posedge clk) begin
        if (write_en) begin
            mem[write_addr] <= write_data;
        end
    end

    // Read
    always @(posedge clk) begin
        read_data <= mem[read_addr];
    end
endmodule
