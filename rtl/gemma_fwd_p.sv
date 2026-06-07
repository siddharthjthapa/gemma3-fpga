`timescale 1ns/1ps
//============================================================================
// gemma_fwd_p - Gemma3-600K forward pass, FULL fp16 datapath, for 100 MHz.
//   Modeled on llama_fwd_p.sv but: fp16 everywhere (pmac16 / fp16_* units),
//   4-lane row-interleaved matmul (64-bit beat = 4 fp16), Gemma RMSNorm
//   (out = x*rsqrt(mean(x^2)+eps)*(1+w)), per-head QK-norm, split-half RoPE,
//   GQA (n_kv_heads=1 -> all q heads share kv head 0), sliding-window
//   attention (window 64), GELU-tanh gated FFN (x/(1+e^-2z), reciprocal =
//   rsqrt^2), post-attention + post-ffn norms, fp16 KV cache, tied classifier.
//   Weights stream from DDR as fp16 (byte addr = halfword_offset << 1).
//============================================================================
module gemma_fwd_p (
    input  logic        clk,
    input  logic        rst,
    input  logic        start,
    input  logic [15:0] token,
    input  logic [15:0] pos,
    output logic        logit_we,
    output logic [9:0]  logit_waddr,
    output logic [15:0] logit_wdata,
    output logic        done,
    output logic [31:0] araddr, output logic [7:0] arlen, output logic [2:0] arsize,
    output logic [1:0]  arburst, output logic arvalid, input logic arready,
    input  logic [63:0] rdata, input logic rvalid, output logic rready,
    input  logic rlast, input logic [1:0] rresp
);
    // ---- dimensions ----
    localparam int DIM=80, HIDDEN=240, NL=7, NH=4, NKVH=1, VOCAB=1024,
                   HS=20, KVD=20, QDIM=80, SEQK=128, KVMUL=4, HALF=10, SW=64,
                   KVSTRIDE=32;  // power-of-2 KV cache stride (addr = shift, not DSP mult)
    // ---- blob halfword offsets (from script/gen_blob.py) ----
    localparam int O_TOK=0, O_RMSA=81920, O_WQ=82480, O_WK=127280, O_WV=138480,
                   O_WO=149680, O_QNORM=194480, O_KNORM=194620, O_POSTATT=194760,
                   O_RMSF=195320, O_W1=195880, O_W2=330280, O_W3=464680,
                   O_POSTFFN=599080, O_RMSFIN=599640, O_FREAL=599720,
                   O_FIMAG=602792, O_WCLS=605864, ROPE_STRIDE=12;
    // RoPE table stride padded to 12 (mult of 4) so each per-position read is
    // 8-byte aligned (real HP aligns ARSIZE=8 addresses down; stride 10 broke
    // odd positions -> shifted RoPE). See script/gen_blob.py.
    // ---- activation RAM layout ----
    localparam int A_X=0, A_XB=80, A_XB2=160, A_Q=240, A_KL=320, A_VL=340,
                   A_HB=360, A_HB2=600, A_ATT=840, ASZ=1096;
    // ---- fp16 constants ----
    localparam logic [15:0] EMBSCALE=16'h4879, INV_N=16'h2266, INV_HS=16'h2A66,
                   INV_SQRT_HS=16'h3328, ONE=16'h3C00, EPS=16'h0000,
                   K2=16'h3E62, K1=16'h29B9, NEG_INF=16'hFC00;
    localparam int LM=3, LA=5, LM1=LM+1, LA1=LA+1;  // +1 for registered act read

    // ---- memories ----
    (* ram_style="block" *)       logic [15:0] kc  [0:NL*SEQK*KVSTRIDE-1];
    (* ram_style="block" *)       logic [15:0] vc  [0:NL*SEQK*KVSTRIDE-1];
    (* ram_style="distributed" *) logic [15:0] act [0:ASZ-1];
    logic [15:0] rwbuf [0:DIM-1], cosb [0:11], sinb [0:11];
    logic [15:0] kc_rdata, vc_rdata;

    // ---- counters / scratch ----
    logic [2:0]  lyr, h, rp_h, qn_h;
    logic [4:0]  rp_i;
    logic [15:0] i, j, t, t2, tt, astart;
    logic [3:0]  cnt;
    logic [15:0] acc, rms_mean, rms_meps, inv2, rms_tmp, rms_g;
    logic [15:0] smax, ssum, at_inv, dexp, etmp;
    logic [15:0] gg, g2, g3, k1g3, inner, gm, gexp, gr, recip, gel;
    logic [15:0] rq0, rq1, rp_q0c, rp_q1s, rp_q0s, rp_q1c;
    logic [15:0] r0, r1, r2, r3;
    logic        is_cls, rp_k;

    logic [15:0] mm_in, mm_out, mm_N, mm_D, rms_in, rms_out, rms_n, rms_invn;
    logic [15:0] res_a, res_b, res_n;
    logic [31:0] mm_woff, wl_off;
    logic [15:0] wl_n;
    logic [15:0] loff_r;
    wire  [15:0] rp_base = rp_k ? 16'(A_KL) : 16'(A_Q + rp_h*HS);
    wire  [15:0] wl_np   = (wl_n + 16'd3) & ~16'd3;

    // ---- AXI read master (fp16) ----
    logic        w_start, w_wide; logic [31:0] w_base, w_nwords; wire w_done;
    wire [15:0]  s_data; wire s_valid; logic s_ready;
    wire [63:0]  w_data; wire w_valid; logic w_ready;
    axi_rd_m16 U_AXI (.clk(clk),.rstn(~rst),.start(w_start),.wide(w_wide),
        .base(w_base),.nwords(w_nwords),.done(w_done),
        .araddr(araddr),.arlen(arlen),.arsize(arsize),.arburst(arburst),.arvalid(arvalid),.arready(arready),
        .rdata(rdata),.rvalid(rvalid),.rready(rready),.rlast(rlast),
        .s_data(s_data),.s_valid(s_valid),.s_ready(s_ready),
        .w_data(w_data),.w_valid(w_valid),.w_ready(w_ready));

    // ---- states ----
    typedef enum logic [6:0] {
        IDLE, EMB_CON, EMB_SC,
        L_ATT_LD, L_ATT_RMS, L_WQ, L_WK, L_WV,
        QN_LDQ, QN_Q0, QN_Q1, QN_LDK, QN_K0,
        ROPE_REQ, COS_CON, SIN_REQ, SIN_CON,
        RP_RD, RP_M0, RP_M1, RP_M2, RP_M3, RP_A0, RP_A1, RP_NEXT,
        L_STKV, SK_WR, L_ATTN, AT_HINIT,
        AS_START, AS_ADDR, AS_FEED, AS_DR, AS_SCL,
        AT_MAX, AT_E0, AT_EXP, AT_EXPW, AT_EA, AT_RS, AT_RSW, AT_RS2, AT_NRM,
        AW_START, AW_ADDR, AW_FEED, AW_DR, AT_HNEXT,
        L_WO, POSTA_LD, POSTA_RMS, RES1,
        FFN_LD, FFN_RMS, L_W1, L_W3, GELU0,
        G_RD, G_S2, G_S3, G_K1, G_IN, G_M, G_EXP, G_EXPW, G_D, G_RS, G_RSW, G_RS2, G_GEL, G_OUT,
        L_W2, POSTF_LD, POSTF_RMS, RES2, L_NEXT,
        FIN_LD, FIN_RMS, CLS, DONE_S,
        MM_START, MM_PRIME, MM_FEED, MM_DR, MM_O0, MM_O1, MM_O2, MM_O3,
        LW_REQ, LW_CON,
        RMS_GO, RA_PRIME, RA_FEED, RA_DR, RMS_S0, RMS_S1, RMS_S2, RMS_S3, RMS_NRM, RMS_NR2,
        RES_RUN
    } state_t;
    state_t st, mmret, rmret, lwret, resret;

    // ---- act read/write ----
    logic [15:0] act_ra1, act_ra2, act_waddr, act_wdata; logic act_we;
    wire  [15:0] act_rd1 = act[act_ra1];
    wire  [15:0] act_rd2 = act[act_ra2];
    always_ff @(posedge clk) if (act_we) act[act_waddr] <= act_wdata;
    // Registered act reads feed the shared elementwise fp units + smax compare,
    // splitting the long st->addr->LUTRAM->operand-mux->fp path at a register
    // (timing closure, like llama's act_rd1_q). Consumers add +1 to their wait.
    logic [15:0] act_rd1_q, act_rd2_q;
    always_ff @(posedge clk) begin act_rd1_q <= act_rd1; act_rd2_q <= act_rd2; end

    // ---- KV cache ----
    wire [15:0] rd_i = (st==AS_FEED) ? (i+16'd1) : i;
    wire [15:0] rd_t = (st==AW_FEED) ? (t+16'd1) : t;
    wire [15:0] kv_raddr  = loff_r + (rd_t*KVSTRIDE) + rd_i; // kvoff=0 (single kv head)
    wire [15:0] kvc_waddr = loff_r + (pos*KVSTRIDE) + i;     // *32 = shift, no DSP
    // Registered KV write: delay enable+address one cycle and use the registered
    // act reads (act_rd1_q/act_rd2_q) so the high-fanout act-LUTRAM read no longer
    // drives the kc/vc BRAM data port directly (was the binding path). Each entry
    // lands one cycle later at its matched address (done well before attention reads).
    logic kvw_q; logic [15:0] kvwa_q;
    always_ff @(posedge clk) begin kvw_q <= (st==SK_WR); kvwa_q <= kvc_waddr; end
    always_ff @(posedge clk) begin
        if (kvw_q) kc[kvwa_q] <= act_rd1_q;
        kc_rdata <= kc[kv_raddr];
    end
    always_ff @(posedge clk) begin
        if (kvw_q) vc[kvwa_q] <= act_rd2_q;
        vc_rdata <= vc[kv_raddr];
    end

    // ---- shared elementwise fp16 units ----
    logic [15:0] mul_a, mul_b, add_a, add_b;
    wire  [15:0] mul_y, add_y;
    fp16_mul_p U_MUL (.clk(clk), .a(mul_a), .b(mul_b), .y(mul_y));
    fp16_add_p U_ADD (.clk(clk), .a(add_a), .b(add_b), .y(add_y));
    logic exps_start, rsqs_start; logic [15:0] exps_x, rsqs_x;
    wire [15:0] exps_y, rsqs_y; wire exps_done, rsqs_done;
    fp16_exp_p   U_EXP (.clk(clk),.rst(rst),.start(exps_start),.x(exps_x),.y(exps_y),.done(exps_done));
    fp16_rsqrt_p U_RSQ (.clk(clk),.rst(rst),.start(rsqs_start),.x(rsqs_x),.y(rsqs_y),.done(rsqs_done));

    // ---- 4 pmac16 lanes (lane 0 also used for scalar dot products) ----
    logic        pm_s0,pm_s1,pm_s2,pm_s3, pm_f0,pm_f1,pm_f2,pm_f3, pm_l0,pm_l1,pm_l2,pm_l3;
    logic [15:0] pm_x0,pm_x1,pm_x2,pm_x3, pm_w0,pm_w1,pm_w2,pm_w3;
    wire         pm_d0,pm_d1,pm_d2,pm_d3; wire [15:0] pm_r0,pm_r1,pm_r2,pm_r3;
    pmac16 PM0 (.clk(clk),.rst(rst),.start(pm_s0),.feed(pm_f0),.x(pm_x0),.w(pm_w0),.last(pm_l0),.done(pm_d0),.result(pm_r0));
    pmac16 PM1 (.clk(clk),.rst(rst),.start(pm_s1),.feed(pm_f1),.x(pm_x1),.w(pm_w1),.last(pm_l1),.done(pm_d1),.result(pm_r1));
    pmac16 PM2 (.clk(clk),.rst(rst),.start(pm_s2),.feed(pm_f2),.x(pm_x2),.w(pm_w2),.last(pm_l2),.done(pm_d2),.result(pm_r2));
    pmac16 PM3 (.clk(clk),.rst(rst),.start(pm_s3),.feed(pm_f3),.x(pm_x3),.w(pm_w3),.last(pm_l3),.done(pm_d3),.result(pm_r3));
    assign w_ready = (st==MM_FEED);

    // classifier logit write stream (4 outputs/group)
    assign logit_we    = is_cls && (st==MM_O0 || st==MM_O1 || st==MM_O2 || st==MM_O3);
    assign logit_waddr = (st==MM_O0)?i[9:0] : (st==MM_O1)?(i[9:0]+10'd1) :
                         (st==MM_O2)?(i[9:0]+10'd2) : (i[9:0]+10'd3);
    assign logit_wdata = (st==MM_O0)?r0 : (st==MM_O1)?r1 : (st==MM_O2)?r2 : r3;

    // ---- pmac feed mux ----
    wire mm_last = w_valid && (j==mm_N-16'd1);
    always_comb begin
        pm_f0=0; pm_x0=act_rd1; pm_w0=act_rd1; pm_l0=0;
        pm_f1=0; pm_x1=act_rd1; pm_w1=w_data[31:16]; pm_l1=0;
        pm_f2=0; pm_x2=act_rd1; pm_w2=w_data[47:32]; pm_l2=0;
        pm_f3=0; pm_x3=act_rd1; pm_w3=w_data[63:48]; pm_l3=0;
        case (st)
            MM_FEED: begin
                pm_f0=w_valid; pm_w0=w_data[15:0];  pm_l0=mm_last;
                pm_f1=w_valid; pm_w1=w_data[31:16]; pm_l1=mm_last;
                pm_f2=w_valid; pm_w2=w_data[47:32]; pm_l2=mm_last;
                pm_f3=w_valid; pm_w3=w_data[63:48]; pm_l3=mm_last;
            end
            RA_FEED: begin pm_f0=1; pm_x0=act_rd1; pm_w0=act_rd1; pm_l0=(j==rms_n-16'd1); end
            AS_FEED: begin pm_f0=1; pm_x0=act_rd1; pm_w0=kc_rdata; pm_l0=(i==HS-1); end
            AW_FEED: begin pm_f0=1; pm_x0=act_rd1; pm_w0=vc_rdata; pm_l0=(t==pos);    end
            default: ;
        endcase
    end

    // ---- fp16 greater-than (for softmax max), on the REGISTERED act read ----
    wire cs = act_rd1_q[15], ms = smax[15];
    wire [14:0] ca = act_rd1_q[14:0], ma = smax[14:0];
    wire at_gt = (cs!=ms) ? (~cs) : (cs ? (ca<ma) : (ca>ma));

    // ---- act read mux ----
    always_comb begin
        act_ra1 = 16'd0; act_ra2 = 16'd0;
        case (st)
            MM_FEED: act_ra1 = mm_in + j;
            RA_FEED: act_ra1 = rms_in + j;
            RMS_NRM: act_ra1 = rms_in + j;
            EMB_SC:  act_ra1 = A_X + i;
            AS_FEED: act_ra1 = A_Q + h*HS + i;
            AT_MAX:  act_ra1 = A_ATT + t2;
            AT_E0:   act_ra1 = A_ATT + tt;
            AT_NRM:  act_ra1 = A_ATT + tt;
            AW_FEED: act_ra1 = A_ATT + (t - astart);
            SK_WR:   begin act_ra1 = A_KL + i; act_ra2 = A_VL + i; end
            RES_RUN: begin act_ra1 = res_a + i; act_ra2 = res_b + i; end
            RP_RD:   begin act_ra1 = rp_base + rp_i; act_ra2 = rp_base + rp_i + HALF; end
            G_RD:    act_ra1 = A_HB + i;
            G_OUT:   act_ra1 = A_HB2 + i;
            default: ;
        endcase
    end

    // ---- act write mux ----
    always_comb begin
        act_we=1'b0; act_waddr=16'd0; act_wdata=16'd0;
        case (st)
            EMB_CON: if (s_valid) begin act_we=1; act_waddr=A_X+i; act_wdata=s_data; end
            EMB_SC:  if (cnt==0) begin act_we=1; act_waddr=A_X+i; act_wdata=mul_y; end
            MM_O0:   if (!is_cls) begin act_we=1; act_waddr=mm_out+i;        act_wdata=r0; end
            MM_O1:   if (!is_cls) begin act_we=1; act_waddr=mm_out+i+16'd1; act_wdata=r1; end
            MM_O2:   if (!is_cls) begin act_we=1; act_waddr=mm_out+i+16'd2; act_wdata=r2; end
            MM_O3:   if (!is_cls) begin act_we=1; act_waddr=mm_out+i+16'd3; act_wdata=r3; end
            RMS_NR2: if (cnt==0) begin act_we=1; act_waddr=rms_out+j; act_wdata=mul_y; end
            RP_A0:   if (cnt==0) begin act_we=1; act_waddr=rp_base+rp_i;      act_wdata=add_y; end
            RP_A1:   if (cnt==0) begin act_we=1; act_waddr=rp_base+rp_i+HALF; act_wdata=add_y; end
            AS_SCL:  if (cnt==0) begin act_we=1; act_waddr=A_ATT+(t-astart); act_wdata=mul_y; end
            AT_EXPW: if (exps_done) begin act_we=1; act_waddr=A_ATT+tt; act_wdata=exps_y; end
            AT_NRM:  if (cnt==0) begin act_we=1; act_waddr=A_ATT+tt; act_wdata=mul_y; end
            AW_DR:   if (pm_d0) begin act_we=1; act_waddr=A_XB+h*HS+i; act_wdata=pm_r0; end
            G_OUT:   if (cnt==0) begin act_we=1; act_waddr=A_HB+i; act_wdata=mul_y; end
            RES_RUN: if (cnt==0) begin act_we=1; act_waddr=res_a+i; act_wdata=add_y; end
            default: ;
        endcase
    end

    // ---- shared mul/add operand mux ----
    always_comb begin
        mul_a=16'd0; mul_b=16'd0; add_a=16'd0; add_b=16'd0;
        case (st)
            EMB_SC:  begin mul_a=act_rd1_q; mul_b=EMBSCALE; end
            RMS_S0:  begin mul_a=acc; mul_b=rms_invn; end
            RMS_S1:  begin add_a=rms_mean; add_b=EPS; end
            RMS_NRM: begin mul_a=inv2; mul_b=act_rd1_q; add_a=ONE; add_b=rwbuf[j]; end
            RMS_NR2: begin mul_a=rms_tmp; mul_b=rms_g; end
            AS_SCL:  begin mul_a=acc; mul_b=INV_SQRT_HS; end
            AT_E0:   begin add_a=act_rd1_q; add_b={~smax[15],smax[14:0]}; end
            AT_EA:   begin add_a=ssum; add_b=etmp; end
            AT_RS2:  begin mul_a=at_inv; mul_b=at_inv; end
            AT_NRM:  begin mul_a=act_rd1_q; mul_b=inv2; end
            RP_M0:   begin mul_a=rq0; mul_b=cosb[rp_i]; end
            RP_M1:   begin mul_a=rq1; mul_b=sinb[rp_i]; end
            RP_M2:   begin mul_a=rq0; mul_b=sinb[rp_i]; end
            RP_M3:   begin mul_a=rq1; mul_b=cosb[rp_i]; end
            RP_A0:   begin add_a=rp_q0c; add_b={~rp_q1s[15],rp_q1s[14:0]}; end
            RP_A1:   begin add_a=rp_q0s; add_b=rp_q1c; end
            RES_RUN: begin add_a=act_rd1_q; add_b=act_rd2_q; end
            G_S2:    begin mul_a=gg; mul_b=gg; end
            G_S3:    begin mul_a=g2; mul_b=gg; end
            G_K1:    begin mul_a=K1; mul_b=g3; end
            G_IN:    begin add_a=gg; add_b=k1g3; end
            G_M:     begin mul_a=K2; mul_b=inner; end
            G_D:     begin add_a=ONE; add_b=gexp; end
            G_RS2:   begin mul_a=gr; mul_b=gr; end
            G_GEL:   begin mul_a=gg; mul_b=recip; end
            G_OUT:   begin mul_a=gel; mul_b=act_rd1_q; end
            default: ;
        endcase
    end

    assign s_ready = (st==EMB_CON)||(st==LW_CON)||(st==COS_CON)||(st==SIN_CON);

    // ---- main FSM ----
    always_ff @(posedge clk) begin
        if (rst) begin
            st<=IDLE; done<=0; w_start<=0; w_wide<=0; exps_start<=0; rsqs_start<=0;
            is_cls<=0; cnt<=0;
            pm_s0<=0; pm_s1<=0; pm_s2<=0; pm_s3<=0;
        end else begin
            done<=0; w_start<=0; exps_start<=0; rsqs_start<=0;
            pm_s0<=0; pm_s1<=0; pm_s2<=0; pm_s3<=0;
            case (st)
            IDLE: if (start) begin w_base<=(O_TOK+token*DIM)<<1; w_nwords<=DIM; w_wide<=0; w_start<=1;
                                   i<=0; lyr<=0; is_cls<=0; st<=EMB_CON; end
            EMB_CON: if (s_valid) begin if (i==DIM-1) begin i<=0; cnt<=LM1[3:0]; st<=EMB_SC; end else i<=i+16'd1; end
            EMB_SC:  if (cnt!=0) cnt<=cnt-4'd1; else begin
                         if (i==DIM-1) begin st<=L_ATT_LD; end
                         else begin i<=i+16'd1; cnt<=LM1[3:0]; end
                     end

            // ---- attention RMSNorm + QKV ----
            L_ATT_LD: begin wl_off<=O_RMSA+lyr*DIM; wl_n<=DIM; lwret<=L_ATT_RMS; st<=LW_REQ; end
            L_ATT_RMS:begin rms_in<=A_X; rms_out<=A_XB; rms_n<=DIM; rms_invn<=INV_N; rmret<=L_WQ; st<=RMS_GO; end
            L_WQ: begin mm_in<=A_XB; mm_out<=A_Q;  mm_woff<=O_WQ+lyr*QDIM*DIM; mm_N<=DIM; mm_D<=QDIM; mmret<=L_WK; st<=MM_START; end
            L_WK: begin mm_in<=A_XB; mm_out<=A_KL; mm_woff<=O_WK+lyr*KVD*DIM;  mm_N<=DIM; mm_D<=KVD;  mmret<=L_WV; st<=MM_START; end
            L_WV: begin mm_in<=A_XB; mm_out<=A_VL; mm_woff<=O_WV+lyr*KVD*DIM;  mm_N<=DIM; mm_D<=KVD;  mmret<=QN_LDQ; st<=MM_START; end

            // ---- per-head QK norm ----
            QN_LDQ: begin wl_off<=O_QNORM+lyr*HS; wl_n<=HS; lwret<=QN_Q0; qn_h<=0; st<=LW_REQ; end
            QN_Q0:  begin rms_in<=A_Q+qn_h*HS; rms_out<=A_Q+qn_h*HS; rms_n<=HS; rms_invn<=INV_HS; rmret<=QN_Q1; st<=RMS_GO; end
            QN_Q1:  if (qn_h==NH-1) st<=QN_LDK; else begin qn_h<=qn_h+3'd1; st<=QN_Q0; end
            QN_LDK: begin wl_off<=O_KNORM+lyr*HS; wl_n<=HS; lwret<=QN_K0; st<=LW_REQ; end
            QN_K0:  begin rms_in<=A_KL; rms_out<=A_KL; rms_n<=HS; rms_invn<=INV_HS; rmret<=ROPE_REQ; st<=RMS_GO; end

            // ---- RoPE (split-half) ----
            ROPE_REQ: begin w_base<=(O_FREAL+pos*ROPE_STRIDE)<<1; w_nwords<=12; w_wide<=0; w_start<=1; i<=0; st<=COS_CON; end
            COS_CON:  if (s_valid) begin if (i<HALF) cosb[i]<=s_data; if (i==11) st<=SIN_REQ; else i<=i+16'd1; end
            SIN_REQ:  begin w_base<=(O_FIMAG+pos*ROPE_STRIDE)<<1; w_nwords<=12; w_wide<=0; w_start<=1; i<=0; st<=SIN_CON; end
            SIN_CON:  if (s_valid) begin if (i<HALF) sinb[i]<=s_data; if (i==11) begin rp_k<=0; rp_h<=0; rp_i<=0; st<=RP_RD; end else i<=i+16'd1; end
            RP_RD:  begin rq0<=act_rd1; rq1<=act_rd2; st<=RP_M0; cnt<=LM[3:0]; end
            RP_M0:  if (cnt!=0) cnt<=cnt-4'd1; else begin rp_q0c<=mul_y; st<=RP_M1; cnt<=LM[3:0]; end
            RP_M1:  if (cnt!=0) cnt<=cnt-4'd1; else begin rp_q1s<=mul_y; st<=RP_M2; cnt<=LM[3:0]; end
            RP_M2:  if (cnt!=0) cnt<=cnt-4'd1; else begin rp_q0s<=mul_y; st<=RP_M3; cnt<=LM[3:0]; end
            RP_M3:  if (cnt!=0) cnt<=cnt-4'd1; else begin rp_q1c<=mul_y; st<=RP_A0; cnt<=LA[3:0]; end
            RP_A0:  if (cnt!=0) cnt<=cnt-4'd1; else begin st<=RP_A1; cnt<=LA[3:0]; end
            RP_A1:  if (cnt!=0) cnt<=cnt-4'd1; else st<=RP_NEXT;
            RP_NEXT: begin
                        if (rp_i < HALF-1) begin rp_i<=rp_i+5'd1; st<=RP_RD; end
                        else begin rp_i<=0;
                            if (rp_k==1'b0) begin
                                if (rp_h<NH-1) begin rp_h<=rp_h+3'd1; st<=RP_RD; end
                                else begin rp_k<=1; rp_h<=0; st<=RP_RD; end
                            end else st<=L_STKV;
                        end
                     end

            // ---- write KV cache ----
            L_STKV: begin loff_r<=lyr*SEQK*KVSTRIDE; i<=0; st<=SK_WR; end
            SK_WR:  if (i==KVD-1) st<=L_ATTN; else i<=i+16'd1;

            // ---- attention ----
            L_ATTN: begin loff_r<=lyr*SEQK*KVSTRIDE; astart<=(pos>=SW)?(pos-SW+16'd1):16'd0; h<=0; st<=AT_HINIT; end
            AT_HINIT: begin t<=astart; st<=AS_START; end
            AS_START: begin i<=0; pm_s0<=1; st<=AS_ADDR; end
            AS_ADDR:  st<=AS_FEED;
            AS_FEED:  if (i==HS-1) st<=AS_DR; else i<=i+16'd1;
            AS_DR:    if (pm_d0) begin acc<=pm_r0; st<=AS_SCL; cnt<=LM[3:0]; end
            AS_SCL:   if (cnt!=0) cnt<=cnt-4'd1; else begin
                          if (t==pos) begin t2<=0; smax<=NEG_INF; st<=AT_MAX; end
                          else begin t<=t+16'd1; st<=AS_START; end
                      end
            // registered compare: act_rd1_q holds score[t2-1], so compare for
            // t2>=1 and run one extra iteration (t2 = 0 .. count).
            AT_MAX:   begin if (t2!=16'd0 && at_gt) smax<=act_rd1_q;
                            if (t2==(pos-astart)+16'd1) begin tt<=0; ssum<=16'd0; st<=AT_E0; cnt<=LA1[3:0]; end
                            else t2<=t2+16'd1; end
            AT_E0:    if (cnt!=0) cnt<=cnt-4'd1; else begin dexp<=add_y; st<=AT_EXP; end
            AT_EXP:   begin exps_x<=dexp; exps_start<=1; st<=AT_EXPW; end
            AT_EXPW:  if (exps_done) begin etmp<=exps_y; st<=AT_EA; cnt<=LA[3:0]; end
            AT_EA:    if (cnt!=0) cnt<=cnt-4'd1; else begin ssum<=add_y;
                          if (tt==(pos-astart)) begin st<=AT_RS; end
                          else begin tt<=tt+16'd1; st<=AT_E0; cnt<=LA1[3:0]; end
                      end
            AT_RS:    begin rsqs_x<=ssum; rsqs_start<=1; st<=AT_RSW; end
            AT_RSW:   if (rsqs_done) begin at_inv<=rsqs_y; st<=AT_RS2; cnt<=LM[3:0]; end
            AT_RS2:   if (cnt!=0) cnt<=cnt-4'd1; else begin inv2<=mul_y; tt<=0; st<=AT_NRM; cnt<=LM1[3:0]; end
            AT_NRM:   if (cnt!=0) cnt<=cnt-4'd1; else begin
                          if (tt==(pos-astart)) begin i<=0; st<=AW_START; end
                          else begin tt<=tt+16'd1; cnt<=LM1[3:0]; end
                      end
            AW_START: begin t<=astart; pm_s0<=1; st<=AW_ADDR; end
            AW_ADDR:  st<=AW_FEED;
            AW_FEED:  if (t==pos) st<=AW_DR; else t<=t+16'd1;
            AW_DR:    if (pm_d0) begin if (i==HS-1) st<=AT_HNEXT; else begin i<=i+16'd1; st<=AW_START; end end
            AT_HNEXT: if (h==NH-1) st<=L_WO; else begin h<=h+3'd1; st<=AT_HINIT; end

            // ---- WO + post-attn norm + residual ----
            L_WO:   begin mm_in<=A_XB; mm_out<=A_XB2; mm_woff<=O_WO+lyr*DIM*QDIM; mm_N<=QDIM; mm_D<=DIM; mmret<=POSTA_LD; st<=MM_START; end
            POSTA_LD: begin wl_off<=O_POSTATT+lyr*DIM; wl_n<=DIM; lwret<=POSTA_RMS; st<=LW_REQ; end
            POSTA_RMS:begin rms_in<=A_XB2; rms_out<=A_XB2; rms_n<=DIM; rms_invn<=INV_N; rmret<=RES1; st<=RMS_GO; end
            RES1:   begin res_a<=A_X; res_b<=A_XB2; res_n<=DIM; i<=0; resret<=FFN_LD; st<=RES_RUN; cnt<=LA1[3:0]; end

            // ---- FFN ----
            FFN_LD: begin wl_off<=O_RMSF+lyr*DIM; wl_n<=DIM; lwret<=FFN_RMS; st<=LW_REQ; end
            FFN_RMS:begin rms_in<=A_X; rms_out<=A_XB; rms_n<=DIM; rms_invn<=INV_N; rmret<=L_W1; st<=RMS_GO; end
            L_W1: begin mm_in<=A_XB; mm_out<=A_HB;  mm_woff<=O_W1+lyr*HIDDEN*DIM; mm_N<=DIM; mm_D<=HIDDEN; mmret<=L_W3; st<=MM_START; end
            L_W3: begin mm_in<=A_XB; mm_out<=A_HB2; mm_woff<=O_W3+lyr*HIDDEN*DIM; mm_N<=DIM; mm_D<=HIDDEN; mmret<=GELU0; st<=MM_START; end
            GELU0: begin i<=0; st<=G_RD; end
            // GELU: hb[i] = gg/(1+e^{-K2*(gg+K1*gg^3)}) * up
            G_RD:  begin gg<=act_rd1; st<=G_S2; cnt<=LM[3:0]; end
            G_S2:  if (cnt!=0) cnt<=cnt-4'd1; else begin g2<=mul_y; st<=G_S3; cnt<=LM[3:0]; end
            G_S3:  if (cnt!=0) cnt<=cnt-4'd1; else begin g3<=mul_y; st<=G_K1; cnt<=LM[3:0]; end
            G_K1:  if (cnt!=0) cnt<=cnt-4'd1; else begin k1g3<=mul_y; st<=G_IN; cnt<=LA[3:0]; end
            G_IN:  if (cnt!=0) cnt<=cnt-4'd1; else begin inner<=add_y; st<=G_M; cnt<=LM[3:0]; end
            G_M:   if (cnt!=0) cnt<=cnt-4'd1; else begin gm<=mul_y; st<=G_EXP; end
            G_EXP: begin exps_x<={~gm[15],gm[14:0]}; exps_start<=1; st<=G_EXPW; end
            G_EXPW:if (exps_done) begin gexp<=exps_y; st<=G_D; cnt<=LA[3:0]; end
            G_D:   if (cnt!=0) cnt<=cnt-4'd1; else begin gexp<=add_y; st<=G_RS; end  // d=1+e (reuse gexp)
            G_RS:  begin rsqs_x<=gexp; rsqs_start<=1; st<=G_RSW; end
            G_RSW: if (rsqs_done) begin gr<=rsqs_y; st<=G_RS2; cnt<=LM[3:0]; end
            G_RS2: if (cnt!=0) cnt<=cnt-4'd1; else begin recip<=mul_y; st<=G_GEL; cnt<=LM[3:0]; end
            G_GEL: if (cnt!=0) cnt<=cnt-4'd1; else begin gel<=mul_y; st<=G_OUT; cnt<=LM1[3:0]; end
            G_OUT: if (cnt!=0) cnt<=cnt-4'd1; else begin
                       if (i==HIDDEN-1) st<=L_W2; else begin i<=i+16'd1; st<=G_RD; cnt<=LM1[3:0]; end
                   end
            L_W2: begin mm_in<=A_HB; mm_out<=A_XB; mm_woff<=O_W2+lyr*DIM*HIDDEN; mm_N<=HIDDEN; mm_D<=DIM; mmret<=POSTF_LD; st<=MM_START; end
            POSTF_LD: begin wl_off<=O_POSTFFN+lyr*DIM; wl_n<=DIM; lwret<=POSTF_RMS; st<=LW_REQ; end
            POSTF_RMS:begin rms_in<=A_XB; rms_out<=A_XB; rms_n<=DIM; rms_invn<=INV_N; rmret<=RES2; st<=RMS_GO; end
            RES2:   begin res_a<=A_X; res_b<=A_XB; res_n<=DIM; i<=0; resret<=L_NEXT; st<=RES_RUN; cnt<=LA1[3:0]; end
            L_NEXT: if (lyr==NL-1) st<=FIN_LD; else begin lyr<=lyr+3'd1; st<=L_ATT_LD; end

            // ---- final norm + classifier ----
            FIN_LD: begin wl_off<=O_RMSFIN; wl_n<=DIM; lwret<=FIN_RMS; st<=LW_REQ; end
            FIN_RMS:begin rms_in<=A_X; rms_out<=A_X; rms_n<=DIM; rms_invn<=INV_N; rmret<=CLS; st<=RMS_GO; end
            CLS:    begin mm_in<=A_X; mm_out<=0; mm_woff<=O_WCLS; mm_N<=DIM; mm_D<=VOCAB; mmret<=DONE_S; is_cls<=1; st<=MM_START; end
            DONE_S: begin done<=1; is_cls<=0; st<=IDLE; end

            // ---- 4-lane interleaved matmul ----
            MM_START: begin w_base<=mm_woff<<1; w_nwords<=mm_N*mm_D; w_wide<=1; w_start<=1;
                            i<=0; j<=0; pm_s0<=1; pm_s1<=1; pm_s2<=1; pm_s3<=1; st<=MM_PRIME; end
            MM_PRIME: st<=MM_FEED;
            MM_FEED: if (w_valid) begin if (j==mm_N-16'd1) begin j<=0; st<=MM_DR; end else j<=j+16'd1; end
            MM_DR:   if (pm_d0) begin r0<=pm_r0; r1<=pm_r1; r2<=pm_r2; r3<=pm_r3; st<=MM_O0; end
            MM_O0:   st<=MM_O1;
            MM_O1:   st<=MM_O2;
            MM_O2:   st<=MM_O3;
            MM_O3:   if (i==mm_D-16'd4) st<=mmret;
                     else begin i<=i+16'd4; pm_s0<=1; pm_s1<=1; pm_s2<=1; pm_s3<=1; st<=MM_PRIME; end

            // ---- weight load into rwbuf ----
            LW_REQ: begin w_base<=wl_off<<1; w_nwords<=wl_np; w_wide<=0; w_start<=1; i<=0; st<=LW_CON; end
            LW_CON: if (s_valid) begin if (i<wl_n) rwbuf[i]<=s_data;
                                       if (i==wl_np-16'd1) st<=lwret; else i<=i+16'd1; end

            // ---- RMSNorm subroutine (Gemma: out = inv*x*(1+w)) ----
            RMS_GO:  begin pm_s0<=1; j<=0; st<=RA_PRIME; end
            RA_PRIME:st<=RA_FEED;
            RA_FEED: if (j==rms_n-16'd1) begin st<=RA_DR; end else j<=j+16'd1;
            RA_DR:   if (pm_d0) begin acc<=pm_r0; st<=RMS_S0; cnt<=LM[3:0]; end
            RMS_S0:  if (cnt!=0) cnt<=cnt-4'd1; else begin rms_mean<=mul_y; st<=RMS_S1; cnt<=LA[3:0]; end
            RMS_S1:  if (cnt!=0) cnt<=cnt-4'd1; else begin rms_meps<=add_y; st<=RMS_S2; end
            RMS_S2:  begin rsqs_x<=rms_meps; rsqs_start<=1; st<=RMS_S3; end
            RMS_S3:  if (rsqs_done) begin inv2<=rsqs_y; j<=0; st<=RMS_NRM; cnt<=LA[3:0]; end
            RMS_NRM: if (cnt!=0) cnt<=cnt-4'd1; else begin rms_tmp<=mul_y; rms_g<=add_y; st<=RMS_NR2; cnt<=LM[3:0]; end
            RMS_NR2: if (cnt!=0) cnt<=cnt-4'd1; else begin
                         if (j==rms_n-16'd1) st<=rmret;
                         else begin j<=j+16'd1; st<=RMS_NRM; cnt<=LA[3:0]; end
                     end

            // ---- residual add: act[res_a+i] += act[res_b+i] ----
            RES_RUN: if (cnt!=0) cnt<=cnt-4'd1; else begin
                         if (i==res_n-16'd1) st<=resret;
                         else begin i<=i+16'd1; cnt<=LA1[3:0]; end
                     end
            default: st<=IDLE;
            endcase
        end
    end

    // synthesis translate_off
    integer nfwd=0, z, fp;
    always @(posedge clk) begin
        if (st==DONE_S) nfwd<=nfwd+1;
        if (nfwd==0 && lyr==0) begin
            if (st==L_WQ)  begin
                fp=$fopen("rtl_emb.txt","w"); for(z=0;z<DIM;z++) $fdisplay(fp,"%04x",act[A_X+z]);  $fclose(fp);
                fp=$fopen("rtl_xb.txt","w");  for(z=0;z<DIM;z++) $fdisplay(fp,"%04x",act[A_XB+z]); $fclose(fp);
            end
            if (st==QN_LDQ) begin
                fp=$fopen("rtl_q.txt","w"); for(z=0;z<QDIM;z++) $fdisplay(fp,"%04x",act[A_Q+z]); $fclose(fp);
            end
            if (st==L_WO)   begin
                fp=$fopen("rtl_attn.txt","w"); for(z=0;z<QDIM;z++) $fdisplay(fp,"%04x",act[A_XB+z]); $fclose(fp);
            end
            if (st==POSTA_LD) begin
                fp=$fopen("rtl_wo.txt","w"); for(z=0;z<DIM;z++) $fdisplay(fp,"%04x",act[A_XB2+z]); $fclose(fp);
            end
            if (st==RES1)   begin
                fp=$fopen("rtl_o.txt","w"); for(z=0;z<DIM;z++) $fdisplay(fp,"%04x",act[A_XB2+z]); $fclose(fp);
            end
            if (st==FFN_LD) begin
                fp=$fopen("rtl_xatt.txt","w"); for(z=0;z<DIM;z++) $fdisplay(fp,"%04x",act[A_X+z]); $fclose(fp);
            end
            if (st==G_RD && i==0) begin
                fp=$fopen("rtl_g.txt","w"); for(z=0;z<HIDDEN;z++) $fdisplay(fp,"%04x",act[A_HB+z]); $fclose(fp);
                fp=$fopen("rtl_u.txt","w"); for(z=0;z<HIDDEN;z++) $fdisplay(fp,"%04x",act[A_HB2+z]); $fclose(fp);
            end
            if (st==L_W2)   begin
                fp=$fopen("rtl_hb.txt","w"); for(z=0;z<HIDDEN;z++) $fdisplay(fp,"%04x",act[A_HB+z]); $fclose(fp);
            end
            if (st==L_NEXT) begin
                fp=$fopen("rtl_xL0.txt","w"); for(z=0;z<DIM;z++) $fdisplay(fp,"%04x",act[A_X+z]); $fclose(fp);
            end
        end
    end
    // synthesis translate_on
endmodule
