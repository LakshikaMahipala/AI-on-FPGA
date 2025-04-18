timescale 1ns / 1ps

module testbench_uart_top;

    reg clk = 0;
    reg rst = 1;
    reg uart_rx_pin;
    wire uart_tx_pin;

    // Instantiate the full system
    uart_top dut (
        .clk(clk),
        .rst(rst),
        .uart_rx_pin(uart_rx_pin),
        .uart_tx_pin(uart_tx_pin)
    );

    // Clock: 50 MHz = 20ns period
    always #10 clk = ~clk;

    // UART task: 115200 baud (≈ 8680 ns per bit)
    task send_uart_byte(input [7:0] byte);
        integer i;
        begin
            uart_rx_pin = 0; #8680; // Start bit
            for (i = 0; i < 8; i = i + 1) begin
                uart_rx_pin = byte[i]; #8680;
            end
            uart_rx_pin = 1; #8680; // Stop bit
        end
    endtask

    initial begin
        uart_rx_pin = 1; // Idle

        #100; rst = 0;

        // Send 9 dummy pixel bytes
        send_uart_byte(8'd10);
        send_uart_byte(8'd20);
        send_uart_byte(8'd30);
        send_uart_byte(8'd40);
        send_uart_byte(8'd50);
        send_uart_byte(8'd60);
        send_uart_byte(8'd70);
        send_uart_byte(8'd80);
        send_uart_byte(8'd90);

        // Wait long enough for processing
        #1000000;
        $finish;
    end

endmodule
