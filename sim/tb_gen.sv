`timescale 1ns/1ps
//============================================================================
// tb_gen - multi-token generation on gemma_fwd_p, decoded to text in the
//   console. Drives the core like main.c: teacher-force the prompt, then
//   greedy-argmax sample; decode each emitted token via the tokenizer tables.
//============================================================================
module tb_gen;
    `include "gen_meta.svh"            // NPROMPT, NOFF, NBYTES
    localparam int WORDS=686760, VOC=1024, STEPS=80, EOS=1;

    logic clk=0, rst=1, start=0, done;
    logic [15:0] tok_r, pos_r;
    logic [15:0] L [0:VOC-1];
    logic [31:0] voff [0:NOFF-1];
    logic [7:0]  vbytes [0:NBYTES-1];
    logic [15:0] ptok [0:NPROMPT-1];
    always #5 clk=~clk;

    logic [31:0] araddr; logic [63:0] rdata;
    logic [7:0] arlen; logic [2:0] arsize; logic [1:0] arburst, rresp;
    logic arvalid, arready, rvalid, rready, rlast;

    ddr_model16 #(.WORDS(WORDS)) u_ddr (
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
    function automatic int argmax();
        int mi; real mp; mi=0; mp=f2r(L[0]);
        for (int k=1;k<VOC;k++) if (f2r(L[k])>mp) begin mp=f2r(L[k]); mi=k; end
        return mi;
    endfunction
    task automatic run(input [15:0] tk, input [15:0] ps);
        @(posedge clk); tok_r=tk; pos_r=ps; start=1;
        @(posedge clk); start=0; wait(done); @(posedge clk);
    endtask
    task automatic emit(input int id);          // print the token's piece bytes
        int s,e; s=voff[id]; e=voff[id+1];
        for (int j=s;j<e;j++) begin
            byte unsigned c; c=vbytes[j];
            if (c==8'h0a || (c>=8'h20 && c<8'h7f)) $write("%c", c);
        end
    endtask

    int tok, nxt, pos;
    initial begin
        $readmemh("model.hex",       u_ddr.mem);
        $readmemh("vocab_off.hex",   voff);
        $readmemh("vocab_bytes.hex", vbytes);
        $readmemh("prompt_toks.hex", ptok);
        repeat(4) @(posedge clk); rst=0;

        $write("\n--- prompt ---\n");
        for (int p=0;p<NPROMPT;p++) emit(ptok[p]);
        $write("\n--- generation ---\n");

        tok = ptok[0]; pos = 0;
        while (pos < STEPS) begin
            run(tok[15:0], pos[15:0]);
            nxt = argmax();
            if (pos < NPROMPT-1) nxt = ptok[pos+1];   // teacher-force the prompt
            pos++;
            if (nxt == EOS) break;
            if (pos >= NPROMPT) emit(nxt);
            tok = nxt;
        end
        $write("\n--- done (%0d tokens) ---\n", pos);
        $finish;
    end
    initial begin #2000000000 $display("TIMEOUT"); $finish; end
endmodule
