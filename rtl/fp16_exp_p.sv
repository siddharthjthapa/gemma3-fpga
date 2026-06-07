`timescale 1ns/1ps
//============================================================================
// fp16_exp_p - e^x for fp16. Same algorithm as llama's fp32_exp_p:
//   k=x*log2e, n=round(k), r=x-n*ln2, degree-5 Horner poly, scale by 2^n via
//   the exponent field. On fp16_mul_p (lat 3)/fp16_add_p (lat 5) + wait-counter.
//   fp16 = 1/5/10, bias 15.  Used for softmax and (via 1/(1+e^-2z)) GELU.
//============================================================================
module fp16_exp_p (
    input  logic        clk,
    input  logic        rst,
    input  logic        start,
    input  logic [15:0] x,
    output logic [15:0] y,
    output logic        done
);
    localparam logic [15:0] LOG2E=16'h3DC5, LN2=16'h398C;
    localparam logic [15:0] C5=16'h2044, C4=16'h2955, C3=16'h3155,
                            C2=16'h3800, C1=16'h3C00, C0=16'h3C00;
    localparam int LM=3, LA=5;

    logic [15:0] mul_a, mul_b, add_a, add_b;
    wire  [15:0] mul_y, add_y;
    fp16_mul_p U_M (.clk(clk), .a(mul_a), .b(mul_b), .y(mul_y));
    fp16_add_p U_A (.clk(clk), .a(add_a), .b(add_b), .y(add_y));

    logic [15:0] k, nf, r, p, tmp;
    logic signed [15:0] n;
    wire signed [15:0] n_w;  fp2int_round16 U_FN (.k(k), .n(n_w));
    wire [15:0]        nf_w; int2fp16       U_NF (.n(n), .y(nf_w));
    logic [2:0] pidx, cnt;

    wire [15:0] coeff = (pidx==0)?C4 : (pidx==1)?C3 : (pidx==2)?C2 : (pidx==3)?C1 : C0;
    logic signed [10:0] newe;

    typedef enum logic [3:0] {IDLE,MUL_K,RD_N,RD_NF,MUL_NL,ADD_R,P_INIT,P_MUL,P_ADD,SCALE,FIN} st_t;
    st_t st;

    always_comb begin
        mul_a=16'd0; mul_b=16'd0; add_a=16'd0; add_b=16'd0;
        case (st)
            MUL_K:  begin mul_a=x;  mul_b=LOG2E; end
            MUL_NL: begin mul_a=nf; mul_b=LN2;   end
            ADD_R:  begin add_a=x;  add_b={~tmp[15],tmp[14:0]}; end
            P_MUL:  begin mul_a=p;  mul_b=r;      end
            P_ADD:  begin add_a=tmp; add_b=coeff; end
            default: ;
        endcase
        newe = $signed({6'b0,p[14:10]}) + n;
    end

    always_ff @(posedge clk) begin
        if (rst) begin st<=IDLE; done<=1'b0; end
        else begin
            done<=1'b0;
            case (st)
                IDLE:   if (start) begin st<=MUL_K; cnt<=LM[2:0]; end
                MUL_K:  if (cnt==0) begin k<=mul_y;  st<=RD_N;  end else cnt<=cnt-3'd1;
                RD_N:   begin n<=n_w;    st<=RD_NF; end
                RD_NF:  begin nf<=nf_w;  st<=MUL_NL; cnt<=LM[2:0]; end
                MUL_NL: if (cnt==0) begin tmp<=mul_y; st<=ADD_R; cnt<=LA[2:0]; end else cnt<=cnt-3'd1;
                ADD_R:  if (cnt==0) begin r<=add_y;   st<=P_INIT; end else cnt<=cnt-3'd1;
                P_INIT: begin p<=C5; pidx<=3'd0; st<=P_MUL; cnt<=LM[2:0]; end
                P_MUL:  if (cnt==0) begin tmp<=mul_y; st<=P_ADD; cnt<=LA[2:0]; end else cnt<=cnt-3'd1;
                P_ADD:  if (cnt==0) begin p<=add_y; if (pidx==3'd4) st<=SCALE;
                                       else begin pidx<=pidx+3'd1; st<=P_MUL; cnt<=LM[2:0]; end end
                        else cnt<=cnt-3'd1;
                SCALE:  begin
                            if      (newe<=11'sd0)   y<=16'd0;
                            else if (newe>=11'sd31)  y<={p[15],5'h1F,10'd0};
                            else                     y<={p[15],newe[4:0],p[9:0]};
                            st<=FIN;
                        end
                FIN:    begin done<=1'b1; st<=IDLE; end
                default: st<=IDLE;
            endcase
        end
    end
endmodule
