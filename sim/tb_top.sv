`timescale 1ns/1ps
//============================================================================
// tb_top - exercise the FULL gemma_top exactly like the PS driver does:
//   AXI-Lite writes (WBASE/TOKEN/POS/CTRL), poll STATUS, then read the 1024
//   logits back through the AXI-Lite BRAM path (0x1000+4k). Weights come from
//   the realistic HP model (ddr_model16_rl). This covers the readback path the
//   direct-core testbenches never touch. Must reproduce argmax 380 then 966.
//============================================================================
module tb_top;
    localparam int WORDS=686760, VOC=1024;
    localparam [12:0] A_CTRL=13'h000, A_STAT=13'h004, A_TOK=13'h008,
                      A_POS=13'h00C, A_WBASE=13'h010, A_LOG=13'h1000;
    logic clk=0, rstn=0; always #5 clk=~clk;

    // AXI-Lite
    logic [12:0] awaddr, araddr_l; logic awvalid, awready, wvalid, wready, bvalid, bready;
    logic [31:0] wdata, rdata_l; logic [3:0] wstrb; logic arvalid, arready, rvalid_l, rready_l;
    logic [1:0] bresp, rresp_l;
    // AXI-HP (master -> ddr)
    logic [31:0] m_araddr; logic [7:0] m_arlen; logic [2:0] m_arsize; logic [1:0] m_arburst;
    logic [3:0] m_arcache; logic [2:0] m_arprot; logic m_arvalid, m_arready;
    logic [63:0] m_rdata; logic [1:0] m_rresp; logic m_rvalid, m_rlast, m_rready;
    logic [31:0] m_awaddr; logic [7:0] m_awlen; logic [2:0] m_awsize; logic [1:0] m_awburst;
    logic [3:0] m_awcache; logic [2:0] m_awprot; logic m_awvalid, m_awready;
    logic [63:0] m_wdata; logic [7:0] m_wstrb; logic m_wlast, m_wvalid, m_wready;
    logic [1:0] m_bresp; logic m_bvalid, m_bready;

    logic [15:0] L [0:VOC-1];

    gemma_top #(.C_AXIL_AW(13)) dut (
        .clk(clk), .resetn(rstn),
        .s_axil_awaddr(awaddr), .s_axil_awprot(3'd0), .s_axil_awvalid(awvalid), .s_axil_awready(awready),
        .s_axil_wdata(wdata), .s_axil_wstrb(wstrb), .s_axil_wvalid(wvalid), .s_axil_wready(wready),
        .s_axil_bresp(bresp), .s_axil_bvalid(bvalid), .s_axil_bready(bready),
        .s_axil_araddr(araddr_l), .s_axil_arprot(3'd0), .s_axil_arvalid(arvalid), .s_axil_arready(arready),
        .s_axil_rdata(rdata_l), .s_axil_rresp(rresp_l), .s_axil_rvalid(rvalid_l), .s_axil_rready(rready_l),
        .m_axi_araddr(m_araddr), .m_axi_arlen(m_arlen), .m_axi_arsize(m_arsize), .m_axi_arburst(m_arburst),
        .m_axi_arcache(m_arcache), .m_axi_arprot(m_arprot), .m_axi_arvalid(m_arvalid), .m_axi_arready(m_arready),
        .m_axi_rdata(m_rdata), .m_axi_rresp(m_rresp), .m_axi_rvalid(m_rvalid), .m_axi_rlast(m_rlast), .m_axi_rready(m_rready),
        .m_axi_awaddr(m_awaddr), .m_axi_awlen(m_awlen), .m_axi_awsize(m_awsize), .m_axi_awburst(m_awburst),
        .m_axi_awcache(m_awcache), .m_axi_awprot(m_awprot), .m_axi_awvalid(m_awvalid), .m_axi_awready(m_awready),
        .m_axi_wdata(m_wdata), .m_axi_wstrb(m_wstrb), .m_axi_wlast(m_wlast), .m_axi_wvalid(m_wvalid), .m_axi_wready(m_wready),
        .m_axi_bresp(m_bresp), .m_axi_bvalid(m_bvalid), .m_axi_bready(m_bready));

    ddr_model16_rl #(.WORDS(WORDS)) u_ddr (
        .clk(clk), .rstn(rstn),
        .araddr(m_araddr), .arlen(m_arlen), .arsize(m_arsize), .arburst(m_arburst),
        .arvalid(m_arvalid), .arready(m_arready),
        .rdata(m_rdata), .rvalid(m_rvalid), .rlast(m_rlast), .rresp(m_rresp), .rready(m_rready));
    assign m_awready=1; assign m_wready=1; assign m_bvalid=1; assign m_bresp=2'd0;

    function automatic real f2r(input logic [15:0] b);
        real mant; int unsigned e; logic [9:0] m; logic s;
        s=b[15]; e=b[14:10]; m=b[9:0]; if (e==0) return 0.0;
        mant=1.0; for (int k=0;k<10;k++) if (m[k]) mant+=2.0**(k-10);
        return (s?-1.0:1.0)*mant*(2.0**(int'(e)-15));
    endfunction
    function automatic int argmax(input logic [15:0] v [0:VOC-1]);
        int mi; real mp; mi=0; mp=f2r(v[0]);
        for (int k=1;k<VOC;k++) if (f2r(v[k])>mp) begin mp=f2r(v[k]); mi=k; end
        return mi;
    endfunction

    task axil_write(input [12:0] a, input [31:0] d);
        @(posedge clk); awaddr<=a; awvalid<=1; wdata<=d; wvalid<=1; wstrb<=4'hF;
        fork
            begin wait(awready); @(posedge clk); awvalid<=0; end
            begin wait(wready);  @(posedge clk); wvalid<=0;  end
        join
        wait(bvalid); bready<=1; @(posedge clk); bready<=0;
    endtask
    task axil_read(input [12:0] a, output [31:0] d);
        @(posedge clk); araddr_l<=a; arvalid<=1;
        wait(arready); @(posedge clk); arvalid<=0;
        wait(rvalid_l); d=rdata_l; rready_l<=1; @(posedge clk); rready_l<=0;
    endtask

    integer k; logic [31:0] s, lv;
    task run_tok(input [15:0] tk, input [15:0] ps);
        axil_write(A_TOK, {16'd0,tk}); axil_write(A_POS, {16'd0,ps});
        axil_write(A_CTRL, 32'd1);
        s=0; while ((s&1)==0) axil_read(A_STAT, s);
        for (k=0;k<VOC;k++) begin axil_read(A_LOG + 4*k, lv); L[k]=lv[15:0]; end
    endtask

    initial begin
        $readmemh("model.hex", u_ddr.mem);
        awvalid=0; wvalid=0; bready=0; arvalid=0; rready_l=0;
        repeat(6) @(posedge clk); rstn=1; repeat(2) @(posedge clk);
        axil_write(A_WBASE, 32'd0);              // weights at ddr offset 0
        run_tok(16'd386, 16'd0);
        $display("pos0: argmax=%0d (exp 380)", argmax(L));
        run_tok(16'd380, 16'd1);
        $display("pos1: argmax=%0d (exp 966)", argmax(L));
        if (argmax(L)==966) $display("RESULT: PASS (full gemma_top path)");
        else $display("RESULT: FAIL (gemma_top/readback bug reproduced)");
        $finish;
    end
    initial begin #400000000 $display("RESULT: TIMEOUT"); $finish; end
endmodule
