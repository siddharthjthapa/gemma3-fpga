`timescale 1ns/1ps
//============================================================================
// pmac16 - pipelined fp16 dot-product / MAC engine (fp16 analog of llama's
//   pmac). product = fp16_mul_p(x,w) (lat 3) accumulated into K=8 round-robin
//   fp16 accumulators via fp16_add_p (lat 5). Same accumulator re-read every 8
//   cycles > 8-cycle mul+add writeback => no hazard. On `last`, drains then
//   tree-reduces 8->1 (reordered fp16 sum). FULL fp16 datapath.
//
//   Protocol: pulse `start` (clears accs). Assert `feed` with (x,w) per term;
//   assert `last` with `feed` on the final term. `done` pulses with `result`.
//============================================================================
module pmac16 (
    input  logic        clk,
    input  logic        rst,
    input  logic        start,
    input  logic        feed,
    input  logic [15:0] x,
    input  logic [15:0] w,
    input  logic        last,
    output logic        done,
    output logic [15:0] result
);
    localparam int LM=3, LA=5;

    logic [15:0] acc [0:7];
    logic [2:0]  tc;

    logic [15:0] mul_a, mul_b, add_a, add_b;
    wire  [15:0] mul_y, add_y;
    fp16_mul_p umul (.clk(clk), .a(mul_a), .b(mul_b), .y(mul_y));
    fp16_add_p uadd (.clk(clk), .a(add_a), .b(add_b), .y(add_y));

    // input pipeline register (matches pmac: hide the x/w read path)
    logic [15:0] x_r, w_r; logic feed_q, last_q;

    logic        vm [1:LM]; logic [2:0] km [1:LM];
    logic        va [1:LA]; logic [2:0] ka [1:LA];
    logic        vr [1:LA]; logic [1:0] kr [1:LA];

    typedef enum logic [3:0] {IDLE, ACC, DRAIN, R1, R1W, R2, R2W, R3, R3W, FIN} st_t;
    st_t st;
    logic [3:0]  drain_cnt;
    logic [15:0] red [0:3];
    logic [2:0]  issue_cnt, cap_cnt;

    always_comb begin
        mul_a = x_r; mul_b = w_r;
        unique case (st)
            R1:      begin add_a = acc[{issue_cnt[1:0],1'b0}]; add_b = acc[{issue_cnt[1:0],1'b1}]; end
            R2:      begin add_a = red[{issue_cnt[0],1'b0}];   add_b = red[{issue_cnt[0],1'b1}];   end
            R3:      begin add_a = red[0];                     add_b = red[1];                     end
            default: begin add_a = acc[km[LM]];                add_b = mul_y;                      end
        endcase
    end

    integer i;
    always_ff @(posedge clk) begin
        if (rst) begin
            st <= IDLE; done <= 1'b0; tc <= 3'd0; feed_q <= 1'b0; last_q <= 1'b0;
            for (i=1;i<=LM;i++) vm[i] <= 1'b0;
            for (i=1;i<=LA;i++) begin va[i] <= 1'b0; vr[i] <= 1'b0; end
        end else begin
            done <= 1'b0;

            x_r <= x; w_r <= w; feed_q <= (st==ACC) && feed; last_q <= last;

            vm[1] <= feed_q; km[1] <= tc;
            for (i=2;i<=LM;i++) begin vm[i] <= vm[i-1]; km[i] <= km[i-1]; end
            if (feed_q) tc <= tc + 3'd1;

            va[1] <= vm[LM]; ka[1] <= km[LM];
            for (i=2;i<=LA;i++) begin va[i] <= va[i-1]; ka[i] <= ka[i-1]; end
            if (va[LA]) acc[ka[LA]] <= add_y;

            vr[1] <= 1'b0; kr[1] <= 2'd0;
            for (i=2;i<=LA;i++) begin vr[i] <= vr[i-1]; kr[i] <= kr[i-1]; end
            if (vr[LA]) begin red[kr[LA]] <= add_y; cap_cnt <= cap_cnt + 3'd1; end

            case (st)
            IDLE: if (start) begin
                      for (i=0;i<8;i++) acc[i] <= 16'd0;
                      tc <= 3'd0; st <= ACC;
                      for (i=1;i<=LM;i++) vm[i] <= 1'b0;
                      for (i=1;i<=LA;i++) va[i] <= 1'b0;
                  end
            ACC: if (feed_q && last_q) begin drain_cnt <= 4'd9; st <= DRAIN; end  // LM+LA+1
            DRAIN: if (drain_cnt==0) begin issue_cnt<=0; cap_cnt<=0; st<=R1; end
                   else drain_cnt <= drain_cnt - 4'd1;

            R1: begin vr[1] <= 1'b1; kr[1] <= issue_cnt[1:0];
                      if (issue_cnt==3'd3) st <= R1W; else issue_cnt <= issue_cnt + 3'd1; end
            R1W: if (cap_cnt==3'd4) begin issue_cnt<=0; cap_cnt<=0; st<=R2; end

            R2: begin vr[1] <= 1'b1; kr[1] <= {1'b0, issue_cnt[0]};
                      if (issue_cnt==3'd1) st <= R2W; else issue_cnt <= issue_cnt + 3'd1; end
            R2W: if (cap_cnt==3'd2) begin issue_cnt<=0; cap_cnt<=0; st<=R3; end

            R3: begin vr[1] <= 1'b1; kr[1] <= 2'd0; st <= R3W; end
            R3W: if (cap_cnt==3'd1) st <= FIN;

            FIN: begin result <= red[0]; done <= 1'b1; st <= IDLE; end
            default: st <= IDLE;
            endcase
        end
    end
endmodule
