`timescale 1ns/1ps
// tb_gemma_rl - same as tb_gemma but with the REALISTIC AXI DDR model
// (ddr_model16_rl: AR latency + R latency + rvalid gaps). If pos0/pos1 argmax
// breaks here while tb_gemma passes, axi_rd_m16 has a handshake bug.
module tb_gemma_rl;
    localparam int WORDS=686760, VOC=1024;
    logic clk=0, rst=1, start=0, done;
    logic [15:0] tok_r, pos_r;
    logic [15:0] L [0:VOC-1], REF0 [0:VOC-1], REF1 [0:VOC-1];
    always #5 clk=~clk;

    logic [31:0] araddr; logic [63:0] rdata;
    logic [7:0]  arlen; logic [2:0] arsize; logic [1:0] arburst, rresp;
    logic        arvalid, arready, rvalid, rready, rlast;

    ddr_model16_rl #(.WORDS(WORDS)) u_ddr (
        .clk(clk), .rstn(~rst),
        .araddr(araddr), .arlen(arlen), .arsize(arsize), .arburst(arburst),
        .arvalid(arvalid), .arready(arready),
        .rdata(rdata), .rvalid(rvalid), .rlast(rlast), .rresp(rresp), .rready(rready));

    logic lg_we; logic [9:0] lg_waddr; logic [15:0] lg_wdata;
    always_ff @(posedge clk) if (lg_we) L[lg_waddr] <= lg_wdata;

    gemma_fwd_p dut (
        .clk(clk), .rst(rst), .start(start), .token(tok_r), .pos(pos_r),
        .logit_we(lg_we), .logit_waddr(lg_waddr), .logit_wdata(lg_wdata), .done(done),
        .araddr(araddr), .arlen(arlen), .arsize(arsize), .arburst(arburst),
        .arvalid(arvalid), .arready(arready),
        .rdata(rdata), .rvalid(rvalid), .rready(rready), .rlast(rlast), .rresp(rresp));

    function automatic real f2r(input logic [15:0] b);
        real mant; int unsigned e; logic [9:0] m; logic s;
        s=b[15]; e=b[14:10]; m=b[9:0];
        if (e==0) return 0.0;
        mant=1.0; for (int k=0;k<10;k++) if (m[k]) mant+=2.0**(k-10);
        return (s?-1.0:1.0)*mant*(2.0**(int'(e)-15));
    endfunction
    function automatic int argmax(input logic [15:0] v [0:VOC-1]);
        int mi; real mp; mi=0; mp=f2r(v[0]);
        for (int k=1;k<VOC;k++) if (f2r(v[k])>mp) begin mp=f2r(v[k]); mi=k; end
        return mi;
    endfunction
    task run(input [15:0] tk, input [15:0] ps);
        @(posedge clk); tok_r=tk; pos_r=ps; start=1;
        @(posedge clk); start=0; wait(done); @(posedge clk);
    endtask

    initial begin
        $readmemh("model.hex", u_ddr.mem);
        $readmemh("logits_fp16.hex", REF0);
        $readmemh("logits_p1_fp16.hex", REF1);
        repeat(4) @(posedge clk); rst=0;
        run(16'd386, 16'd0);
        $display("pos0: argmax=%0d (exp 380)", argmax(L));
        run(16'd380, 16'd1);
        $display("pos1: argmax=%0d (exp 966)", argmax(L));
        if (argmax(L)==966) $display("RESULT: PASS (argmax)");
        else $display("RESULT: FAIL (realistic-AXI bug reproduced)");
        $finish;
    end
    initial begin #300000000 $display("RESULT: TIMEOUT"); $finish; end
endmodule
