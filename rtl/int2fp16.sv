`timescale 1ns/1ps
//============================================================================
// int2fp16 - convert small signed integer to fp16 (exact for |n| <= 2048).
//   Used by exp to form 2^n's exponent path. fp16 = 1/5/10, bias 15.
//============================================================================
module int2fp16 (
    input  logic signed [15:0] n,
    output logic [15:0]        y
);
    logic        sign;
    logic [15:0] mag;
    logic [4:0]  msb;
    logic [4:0]  e;
    logic [31:0] sh;
    logic [9:0]  frac;
    integer i;
    always_comb begin
        if (n == 16'sd0) begin
            y = 16'd0;
        end else begin
            sign = n[15];
            mag  = sign ? (~n + 16'sd1) : n;       // absolute value
            msb  = 5'd0;
            for (i = 0; i < 16; i = i + 1) if (mag[i]) msb = i[4:0];
            e    = 5'd15 + msb;
            sh   = {16'd0, mag} << (5'd10 - msb);  // place leading 1 at bit 10
            frac = sh[9:0];
            y    = {sign, e, frac};
        end
    end
endmodule
