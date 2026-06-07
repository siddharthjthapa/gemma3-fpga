`timescale 1ns/1ps
//============================================================================
// fp16_add_p - 5-stage pipelined IEEE-754 half-precision add/sub.
//   Latency = 5, 1 result/cycle. fp16 = 1/5/10 (bias 15). Round-to-nearest-
//   even, subnormals flushed. Structurally identical to fp32_add_p, narrowed:
//     A unpack+order | B align | C add/sub | D normalize(LZC) | E round+pack
//============================================================================
module fp16_add_p (
    input  logic        clk,
    input  logic [15:0] a,
    input  logic [15:0] b,
    output logic [15:0] y
);
    // ---------- Stage A: unpack + magnitude order ----------
    wire        sa=a[15], sb=b[15];
    wire [4:0]  ea=a[14:10], eb=b[14:10];
    wire [10:0] siga = (ea==5'd0)?11'd0:{1'b1,a[9:0]};
    wire [10:0] sigb = (eb==5'd0)?11'd0:{1'b1,b[9:0]};
    wire        a_ge = (ea>eb)||((ea==eb)&&(a[9:0]>=b[9:0]));
    wire [4:0]  eB_a = a_ge?ea:eb;
    wire [10:0] sB_a = a_ge?siga:sigb;
    wire        gB_a = a_ge?sa:sb;
    wire [10:0] sS_a = a_ge?sigb:siga;
    wire        gS_a = a_ge?sb:sa;
    wire [4:0]  eS_a = a_ge?eb:ea;
    wire [4:0]  dexp_a = eB_a - eS_a;

    logic        gB_A, gS_A; logic [4:0] eB_A, dexp_A; logic [10:0] sB_A, sS_A;
    always_ff @(posedge clk) begin
        gB_A<=gB_a; gS_A<=gS_a; eB_A<=eB_a; dexp_A<=dexp_a; sB_A<=sB_a; sS_A<=sS_a;
    end

    // ---------- Stage B: align the smaller significand ----------
    wire [4:0]  sh_b    = (dexp_A>5'd13)?5'd13:dexp_A;
    wire [13:0] bigA_b  = {sB_A,3'b000};
    wire [13:0] smlE_b  = {sS_A,3'b000};
    wire [13:0] smlSh_b = smlE_b >> sh_b;
    wire [13:0] mask_b  = (sh_b==5'd0)?14'd0:((14'd1<<sh_b)-14'd1);
    wire        stky_b  = |(smlE_b & mask_b);
    wire [13:0] smlAln_b= smlSh_b | {13'd0,stky_b};
    wire        same_b  = (gB_A==gS_A);

    logic       gB_B, same_B; logic [4:0] eB_B; logic [13:0] bigA_B, smlAln_B;
    always_ff @(posedge clk) begin
        gB_B<=gB_A; same_B<=same_b; eB_B<=eB_A; bigA_B<=bigA_b; smlAln_B<=smlAln_b;
    end

    // ---------- Stage C: add / subtract ----------
    wire [14:0] sum_c = same_B ? ({1'b0,bigA_B}+{1'b0,smlAln_B})
                               : ({1'b0,bigA_B}-{1'b0,smlAln_B});
    logic gB_C; logic [4:0] eB_C; logic [14:0] sum_C;
    always_ff @(posedge clk) begin gB_C<=gB_B; eB_C<=eB_B; sum_C<=sum_c; end

    // ---------- Stage D: normalize (leading-one) ----------
    logic [14:0] norm_d; logic signed [9:0] eres_d; logic [4:0] shamt_d; integer id;
    always_comb begin
        shamt_d=5'd0;
        if (sum_C[14]) begin norm_d=sum_C>>1; eres_d=$signed({5'b0,eB_C})+10'sd1; end
        else if (sum_C==15'd0) begin norm_d=15'd0; eres_d=10'sd0; end
        else begin
            for (id=0; id<14; id=id+1) if (sum_C[id]) shamt_d=5'd13-id[4:0];
            norm_d=sum_C<<shamt_d; eres_d=$signed({5'b0,eB_C})-$signed({5'b0,shamt_d});
        end
    end
    logic gB_D, zero_D; logic [14:0] norm_D; logic signed [9:0] eres_D;
    always_ff @(posedge clk) begin
        gB_D<=gB_C; zero_D<=(sum_C==15'd0); norm_D<=norm_d; eres_D<=eres_d;
    end

    // ---------- Stage E: round-to-nearest-even + pack ----------
    wire [10:0] mant_e = norm_D[13:3];
    wire g_e=norm_D[2], r_e=norm_D[1], s_e=norm_D[0];
    wire roundup_e = g_e & (r_e | s_e | mant_e[0]);
    wire [11:0] mant_r_e = {1'b0,mant_e}+(roundup_e?12'd1:12'd0);
    logic [10:0] mant_f_e; logic signed [9:0] eres2_e;
    always_comb begin
        if (mant_r_e[11]) begin mant_f_e=mant_r_e[11:1]; eres2_e=eres_D+10'sd1; end
        else             begin mant_f_e=mant_r_e[10:0]; eres2_e=eres_D;          end
    end
    logic [15:0] y_e;
    always_comb begin
        if (zero_D)                y_e=16'd0;
        else if (eres2_e>=10'sd31) y_e={gB_D,5'h1F,10'd0};
        else if (eres2_e<=10'sd0)  y_e={gB_D,15'd0};
        else                       y_e={gB_D,eres2_e[4:0],mant_f_e[9:0]};
    end
    always_ff @(posedge clk) y <= y_e;
endmodule
