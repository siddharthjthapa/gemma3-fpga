`timescale 1ns/1ps
//============================================================================
// fp2int_round16 - round fp16 k to nearest signed integer (range used by exp).
//   fp16 = 1/5/10, bias 15.  |k|<0.5 -> 0; [0.5,1) -> 1; else shift the
//   11-bit significand by (exp-15) about bit 10, with a round bit.
//============================================================================
module fp2int_round16 (
    input  logic [15:0]        k,
    output logic signed [15:0] n
);
    logic        ksign;
    logic [4:0]  ke;
    logic [10:0] km;
    logic [5:0]  sh;
    logic [15:0] mag;
    always_comb begin
        ksign = k[15];
        ke    = k[14:10];
        km    = {1'b1, k[9:0]};
        mag   = 16'd0;
        if (ke < 5'd14) begin
            mag = 16'd0;                       // |k| < 0.5
        end else if (ke == 5'd14) begin
            mag = 16'd1;                       // [0.5, 1) -> 1
        end else begin
            sh = (ke - 5'd15);                 // exponent of k (>= 0 here)
            if (sh >= 6'd10) begin
                mag = {5'd0, km} << (sh - 6'd10);
            end else begin
                mag = {5'd0, km} >> (6'd10 - sh);
                if (km[(6'd10 - sh) - 6'd1]) mag = mag + 16'd1;  // round bit
            end
        end
        n = ksign ? -$signed(mag) : $signed(mag);
    end
endmodule
