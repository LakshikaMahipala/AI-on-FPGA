// Stress-detector CNN inference core: conv3x3 -> ReLU/requant -> maxpool2x2 -> FC -> argmax.
//
// One multiply-accumulate per clock cycle (a single DSP slice), fully
// sequential. Bit-exact with golden/cnn_ref.py.
//
// Usage:
//   1. Write the IMG*IMG grayscale pixels (uint8, row-major) through img_we/img_addr/img_data.
//   2. Pulse `start` for one cycle while `busy` is low.
//   3. `done` pulses when `class_out` (0 = not stressed, 1 = stressed) is valid.
//
// Latency (cycles after `start` is sampled until `done`):
//   C_OUT*P*P*4*(K*K+1)  conv/pool   +   N_CLASSES*(N_FEAT+1)  FC   +   1
//   = 88,875 cycles for the default sizes (1.8 ms at 50 MHz).
//
// Memories use combinational reads, so they map to distributed (LUT) RAM.
// To use block RAM, register the read addresses and add one pipeline stage.
module cnn_core #(
    parameter int IMG       = 48,
    parameter int K         = 3,
    parameter int C_OUT     = 4,
    parameter int N_CLASSES = 2,
    parameter int SHIFT     = 8,
    parameter     W_CONV_FILE = "vectors/w_conv.hex",  // C_OUT*K*K int8,  [c][ky][kx]
    parameter     B_CONV_FILE = "vectors/b_conv.hex",  // C_OUT int32
    parameter     W_FC_FILE   = "vectors/w_fc.hex",    // N_CLASSES*N_FEAT int8, [j][c][py][px]
    parameter     B_FC_FILE   = "vectors/b_fc.hex"     // N_CLASSES int32
) (
    input  logic        clk,
    input  logic        rst,
    // image write port
    input  logic        img_we,
    input  logic [$clog2(IMG*IMG)-1:0] img_addr,
    input  logic [7:0]  img_data,
    // control
    input  logic        start,
    output logic        busy,
    output logic        done,
    output logic [7:0]  class_out
);
    localparam int CONV_OUT = IMG - K + 1;
    localparam int P        = CONV_OUT / 2;        // pooled map size
    localparam int N_FEAT   = C_OUT * P * P;

    // ---------------- memories ----------------
    logic [7:0]         img_ram [IMG*IMG];
    logic signed [7:0]  feat_ram[N_FEAT];
    logic signed [7:0]  w_conv  [C_OUT*K*K];
    logic signed [31:0] b_conv  [C_OUT];
    logic signed [7:0]  w_fc    [N_CLASSES*N_FEAT];
    logic signed [31:0] b_fc    [N_CLASSES];

    initial begin
        $readmemh(W_CONV_FILE, w_conv);
        $readmemh(B_CONV_FILE, b_conv);
        $readmemh(W_FC_FILE, w_fc);
        $readmemh(B_FC_FILE, b_fc);
    end

    always_ff @(posedge clk) if (img_we) img_ram[img_addr] <= img_data;

    // ---------------- state ----------------
    typedef enum logic [2:0] {IDLE, CONV_MAC, CONV_ACT, FC_MAC, FC_STORE, FINISH} state_t;
    state_t state;

    int c, py, px, wy, wx, ky, kx;   // conv / pool loop indices
    int j, i;                        // FC loop indices
    logic signed [31:0] acc;
    logic signed [7:0]  pool_max;
    logic signed [31:0] logits[N_CLASSES];
    int best;                        // argmax of logits (lowest index wins ties)

    always_comb begin
        best = 0;
        for (int n = 1; n < N_CLASSES; n++)
            if (logits[n] > logits[best]) best = n;
    end

    // Current conv input pixel, converted uint8 -> int8 (pixel - 128) by flipping the MSB.
    logic [7:0]        pix;
    logic signed [7:0] x_cur, w_cur;
    assign pix   = img_ram[(2*py + wy + ky) * IMG + (2*px + wx + kx)];
    assign x_cur = {~pix[7], pix[6:0]};
    assign w_cur = w_conv[(c*K + ky)*K + kx];

    // ReLU + requantization of the finished conv sum: clip(round(acc / 2^SHIFT), 0, 127)
    logic signed [31:0] rounded;
    logic signed [7:0]  act, new_max;
    always_comb begin
        rounded = (acc + (32'sd1 <<< (SHIFT - 1))) >>> SHIFT;
        if (rounded < 0)        act = 8'sd0;
        else if (rounded > 127) act = 8'sd127;
        else                    act = rounded[7:0];
        new_max = (act > pool_max) ? act : pool_max;
    end

    assign busy = (state != IDLE);

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            done  <= 1'b0;
        end else begin
            done <= 1'b0;
            case (state)
                IDLE: if (start) begin
                    c <= 0; py <= 0; px <= 0; wy <= 0; wx <= 0; ky <= 0; kx <= 0;
                    acc      <= b_conv[0];
                    pool_max <= 8'sd0;          // ReLU outputs are >= 0
                    state    <= CONV_MAC;
                end

                // K*K multiply-accumulates for one conv output
                CONV_MAC: begin
                    acc <= acc + x_cur * w_cur;
                    if (kx == K - 1) begin
                        kx <= 0;
                        if (ky == K - 1) begin ky <= 0; state <= CONV_ACT; end
                        else ky <= ky + 1;
                    end else kx <= kx + 1;
                end

                // Activate, fold into the 2x2 max, move to the next window / pooled output
                CONV_ACT: begin
                    if (wx == 0) begin
                        wx <= 1; acc <= b_conv[c]; pool_max <= new_max; state <= CONV_MAC;
                    end else if (wy == 0) begin
                        wx <= 0; wy <= 1; acc <= b_conv[c]; pool_max <= new_max; state <= CONV_MAC;
                    end else begin
                        // all four window positions done: store the pooled feature
                        feat_ram[(c*P + py)*P + px] <= new_max;
                        wx <= 0; wy <= 0; pool_max <= 8'sd0;
                        if (px < P - 1) begin
                            px <= px + 1; acc <= b_conv[c]; state <= CONV_MAC;
                        end else if (py < P - 1) begin
                            px <= 0; py <= py + 1; acc <= b_conv[c]; state <= CONV_MAC;
                        end else if (c < C_OUT - 1) begin
                            px <= 0; py <= 0; c <= c + 1; acc <= b_conv[c + 1]; state <= CONV_MAC;
                        end else begin
                            j <= 0; i <= 0; acc <= b_fc[0]; state <= FC_MAC;
                        end
                    end
                end

                // Fully connected: N_FEAT multiply-accumulates per class
                FC_MAC: begin
                    acc <= acc + feat_ram[i] * w_fc[j*N_FEAT + i];
                    if (i == N_FEAT - 1) state <= FC_STORE;
                    else i <= i + 1;
                end

                FC_STORE: begin
                    logits[j] <= acc;
                    if (j == N_CLASSES - 1) state <= FINISH;
                    else begin j <= j + 1; i <= 0; acc <= b_fc[j + 1]; state <= FC_MAC; end
                end

                // Argmax (computed combinationally above)
                FINISH: begin
                    class_out <= best[7:0];
                    done  <= 1'b1;
                    state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end
endmodule
