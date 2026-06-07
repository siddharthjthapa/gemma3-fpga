`timescale 1ns/1ps
//============================================================================
// fp16_mul_p - 3-stage pipelined IEEE-754 half-precision multiply.
//   Latency = 3, 1 result/cycle. fp16 = 1 sign / 5 exp (bias 15) / 10 mantissa.
//   Subnormals flushed to zero (exp==0 -> 0), round-to-nearest (guard only),
//   structurally identical to llama's fp32_mul_p with narrowed fields.
//============================================================================
module fp16_mul_p (
    input  logic        clk,
    input  logic [15:0] a,
    input  logic [15:0] b,
    output logic [15:0] y
);
    // ---------- Stage A: unpack ----------
    wire [4:0]  ea=a[14:10], eb=b[14:10];
    wire [9:0]  ma=a[9:0],   mb=b[9:0];
    wire        sy_a = a[15]^b[15];
    wire [10:0] siga_a = {1'b1,ma}, sigb_a = {1'b1,mb};
    wire signed [7:0] esum_a = $signed({3'b0,ea})+$signed({3'b0,eb})-8'sd15;

    logic sy_A, z_A; logic [10:0] siga_A, sigb_A; logic signed [7:0] esum_A;
    always_ff @(posedge clk) begin
        sy_A<=sy_a; z_A<=((ea==5'd0)|(eb==5'd0)); siga_A<=siga_a; sigb_A<=sigb_a; esum_A<=esum_a;
    end

    // ---------- Stage B: 11x11 multiply ----------
    wire [21:0] prod_b = siga_A * sigb_A;
    logic sy_B, z_B; logic [21:0] prod_B; logic signed [7:0] esum_B;
    always_ff @(posedge clk) begin sy_B<=sy_A; z_B<=z_A; prod_B<=prod_b; esum_B<=esum_A; end

    // ---------- Stage C: normalize + round + pack ----------
    logic [9:0] mant_n; logic guard; logic signed [7:0] exp_n;
    always_comb begin
        if (prod_B[21]) begin mant_n=prod_B[20:11]; guard=prod_B[10]; exp_n=esum_B+8'sd1; end
        else            begin mant_n=prod_B[19:10]; guard=prod_B[9];  exp_n=esum_B;        end
    end
    wire [10:0]        mant_r = {1'b0,mant_n}+(guard?11'd1:11'd0);
    wire signed [7:0]  exp_r  = mant_r[10]?(exp_n+8'sd1):exp_n;
    logic [15:0] y_c;
    always_comb begin
        if (z_B)                  y_c={sy_B,15'd0};
        else if (exp_r>=8'sd31)   y_c={sy_B,5'h1F,10'd0};
        else if (exp_r<=8'sd0)    y_c={sy_B,15'd0};
        else                      y_c={sy_B,exp_r[4:0],mant_r[9:0]};
    end
    always_ff @(posedge clk) y <= y_c;
endmodule
