`timescale 1ns/1ps
//============================================================================
// fp16_rsqrt_p - 1/sqrt(x) for fp16 via fast-inverse-sqrt seed + 3 Newton
//   iterations, on fp16_mul_p (lat 3) / fp16_add_p (lat 5) with a wait-counter.
//   fp16 analog of llama's fp32_rsqrt_p. magic 0x59BA (worst rel err ~2e-4).
//   Reciprocal 1/x is obtained elsewhere as rsqrt(x)^2 (no divide unit).
//============================================================================
module fp16_rsqrt_p (
    input  logic        clk,
    input  logic        rst,
    input  logic        start,
    input  logic [15:0] x,
    output logic [15:0] y,
    output logic        done
);
    localparam logic [15:0] HALF=16'h3800, ONE5=16'h3E00, MAGIC=16'h59BA;
    localparam int LM=3, LA=5;

    logic [15:0] mul_a, mul_b, add_a, add_b;
    wire  [15:0] mul_y, add_y;
    fp16_mul_p U_M (.clk(clk), .a(mul_a), .b(mul_b), .y(mul_y));
    fp16_add_p U_A (.clk(clk), .a(add_a), .b(add_b), .y(add_y));

    logic [15:0] y2, xy2, hxy2, tt;
    logic [1:0]  iter;
    logic [2:0]  cnt;

    typedef enum logic [3:0] {IDLE,I_Y2,I_XY2,I_HXY2,I_T,I_YN,FIN} st_t;
    st_t st;

    always_comb begin
        mul_a=16'd0; mul_b=16'd0; add_a=16'd0; add_b=16'd0;
        case (st)
            I_Y2:   begin mul_a=y;   mul_b=y;    end
            I_XY2:  begin mul_a=x;   mul_b=y2;   end
            I_HXY2: begin mul_a=xy2; mul_b=HALF; end
            I_T:    begin add_a=ONE5; add_b={~hxy2[15],hxy2[14:0]}; end
            I_YN:   begin mul_a=y;   mul_b=tt;   end
            default: ;
        endcase
    end

    always_ff @(posedge clk) begin
        if (rst) begin st<=IDLE; done<=1'b0; end
        else begin
            done<=1'b0;
            case (st)
                IDLE:   if (start) begin y<=MAGIC - (x>>1); iter<=2'd0; st<=I_Y2; cnt<=LM[2:0]; end
                I_Y2:   if (cnt==0) begin y2<=mul_y;   st<=I_XY2;  cnt<=LM[2:0]; end else cnt<=cnt-3'd1;
                I_XY2:  if (cnt==0) begin xy2<=mul_y;  st<=I_HXY2; cnt<=LM[2:0]; end else cnt<=cnt-3'd1;
                I_HXY2: if (cnt==0) begin hxy2<=mul_y; st<=I_T;    cnt<=LA[2:0]; end else cnt<=cnt-3'd1;
                I_T:    if (cnt==0) begin tt<=add_y;   st<=I_YN;   cnt<=LM[2:0]; end else cnt<=cnt-3'd1;
                I_YN:   if (cnt==0) begin y<=mul_y; if (iter==2'd2) st<=FIN;
                                       else begin iter<=iter+2'd1; st<=I_Y2; cnt<=LM[2:0]; end end
                        else cnt<=cnt-3'd1;
                FIN:    begin done<=1'b1; st<=IDLE; end
                default: st<=IDLE;
            endcase
        end
    end
endmodule
