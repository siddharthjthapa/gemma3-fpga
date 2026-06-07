`timescale 1ns/1ps
//============================================================================
// gemma_top - SoC wrapper for gemma_fwd_p (fp16 Gemma3-600K core).
//   * AXI4-Lite SLAVE  (S_AXI_LITE): PS control + status + 1024 logit reads
//   * AXI4   read MASTER (M_AXI)    : streams fp16 weights from PS DDR (HP0)
//
// AXI-Lite register map (byte offsets, 8 KB space -> covers 0x1000 logits):
//   0x000 CTRL   (W)  bit0: write 1 -> pulse start
//   0x004 STATUS (R)  bit0: done(latched)   bit1: busy
//   0x008 TOKEN  (RW) [15:0] input token id
//   0x00C POS    (RW) [15:0] sequence position
//   0x010 WBASE  (RW) DDR byte base of the fp16 weight blob
//   0x1000.. LOGITS (R) fp16 logit[k] in low 16 bits at 0x1000 + 4*k, k=0..1023
//============================================================================
module gemma_top #(
    parameter int C_AXIL_AW = 13,    // 8 KB AXI-Lite (covers 0x1000 logits, 1024 words)
    parameter int C_M_AW    = 32,
    parameter int C_M_DW    = 64
) (
    input  logic                  clk,
    input  logic                  resetn,

    // ---------------- AXI4-Lite slave (control) ----------------
    input  logic [C_AXIL_AW-1:0]  s_axil_awaddr,
    input  logic [2:0]            s_axil_awprot,
    input  logic                  s_axil_awvalid,
    output logic                  s_axil_awready,
    input  logic [31:0]           s_axil_wdata,
    input  logic [3:0]            s_axil_wstrb,
    input  logic                  s_axil_wvalid,
    output logic                  s_axil_wready,
    output logic [1:0]            s_axil_bresp,
    output logic                  s_axil_bvalid,
    input  logic                  s_axil_bready,
    input  logic [C_AXIL_AW-1:0]  s_axil_araddr,
    input  logic [2:0]            s_axil_arprot,
    input  logic                  s_axil_arvalid,
    output logic                  s_axil_arready,
    output logic [31:0]           s_axil_rdata,
    output logic [1:0]            s_axil_rresp,
    output logic                  s_axil_rvalid,
    input  logic                  s_axil_rready,

    // ---------------- AXI4 read master (weights) ----------------
    output logic [C_M_AW-1:0]     m_axi_araddr,
    output logic [7:0]            m_axi_arlen,
    output logic [2:0]            m_axi_arsize,
    output logic [1:0]            m_axi_arburst,
    output logic [3:0]            m_axi_arcache,
    output logic [2:0]            m_axi_arprot,
    output logic                  m_axi_arvalid,
    input  logic                  m_axi_arready,
    input  logic [C_M_DW-1:0]     m_axi_rdata,
    input  logic [1:0]            m_axi_rresp,
    input  logic                  m_axi_rvalid,
    input  logic                  m_axi_rlast,
    output logic                  m_axi_rready,
    output logic [C_M_AW-1:0]     m_axi_awaddr,
    output logic [7:0]            m_axi_awlen,
    output logic [2:0]            m_axi_awsize,
    output logic [1:0]            m_axi_awburst,
    output logic [3:0]            m_axi_awcache,
    output logic [2:0]            m_axi_awprot,
    output logic                  m_axi_awvalid,
    input  logic                  m_axi_awready,
    output logic [C_M_DW-1:0]     m_axi_wdata,
    output logic [C_M_DW/8-1:0]   m_axi_wstrb,
    output logic                  m_axi_wlast,
    output logic                  m_axi_wvalid,
    input  logic                  m_axi_wready,
    input  logic [1:0]            m_axi_bresp,
    input  logic                  m_axi_bvalid,
    output logic                  m_axi_bready
);
    wire rst = ~resetn;

    logic        start_pulse;
    logic [15:0] r_token, r_pos;
    logic [31:0] r_wbase;
    logic        busy, done_latched;

    logic        core_done;
    logic        lg_we;  logic [9:0] lg_waddr;  logic [15:0] lg_wdata;
    (* ram_style="block" *) logic [15:0] logitmem [0:1023];
    always_ff @(posedge clk) if (lg_we) logitmem[lg_waddr] <= lg_wdata;
    logic [C_M_AW-1:0] core_araddr;

    gemma_fwd_p u_core (
        .clk(clk), .rst(rst), .start(start_pulse),
        .token(r_token), .pos(r_pos),
        .logit_we(lg_we), .logit_waddr(lg_waddr), .logit_wdata(lg_wdata),
        .done(core_done),
        .araddr(core_araddr), .arlen(m_axi_arlen), .arsize(m_axi_arsize),
        .arburst(m_axi_arburst), .arvalid(m_axi_arvalid), .arready(m_axi_arready),
        .rdata(m_axi_rdata), .rvalid(m_axi_rvalid), .rready(m_axi_rready),
        .rlast(m_axi_rlast), .rresp(m_axi_rresp)
    );

    assign m_axi_araddr  = core_araddr + r_wbase;
    assign m_axi_arcache = 4'b0011;
    assign m_axi_arprot  = 3'b000;
    assign m_axi_awaddr=0; assign m_axi_awlen=0; assign m_axi_awsize=0;
    assign m_axi_awburst=0; assign m_axi_awcache=0; assign m_axi_awprot=0;
    assign m_axi_awvalid=0; assign m_axi_wdata=0; assign m_axi_wstrb=0;
    assign m_axi_wlast=0; assign m_axi_wvalid=0; assign m_axi_bready=1'b1;

    always_ff @(posedge clk) begin
        if (rst) begin busy<=1'b0; done_latched<=1'b0; end
        else begin
            if (start_pulse)    begin busy<=1'b1; done_latched<=1'b0; end
            else if (core_done) begin busy<=1'b0; done_latched<=1'b1; end
        end
    end

    //========================================================================
    // AXI4-Lite slave
    //========================================================================
    localparam logic [C_AXIL_AW-1:0] A_CTRL=13'h000, A_STAT=13'h004,
               A_TOK=13'h008, A_POS=13'h00C, A_WBASE=13'h010;

    assign s_axil_bresp = 2'b00;
    logic [C_AXIL_AW-1:0] awaddr_q;
    logic [31:0]          wdata_q;
    logic                 aw_hs, w_hs;

    always_ff @(posedge clk) begin
        if (rst) begin
            s_axil_awready<=1'b0; s_axil_wready<=1'b0; s_axil_bvalid<=1'b0;
            aw_hs<=1'b0; w_hs<=1'b0;
            r_token<=16'd0; r_pos<=16'd0; r_wbase<=32'd0; start_pulse<=1'b0;
        end else begin
            start_pulse <= 1'b0;
            if (s_axil_awvalid && !aw_hs) begin
                s_axil_awready<=1'b1; awaddr_q<=s_axil_awaddr; aw_hs<=1'b1;
            end else s_axil_awready<=1'b0;
            if (s_axil_wvalid && !w_hs) begin
                s_axil_wready<=1'b1; wdata_q<=s_axil_wdata; w_hs<=1'b1;
            end else s_axil_wready<=1'b0;
            if (aw_hs && w_hs && !s_axil_bvalid) begin
                case (awaddr_q)
                    A_CTRL:  if (wdata_q[0]) start_pulse<=1'b1;
                    A_TOK:   r_token<=wdata_q[15:0];
                    A_POS:   r_pos  <=wdata_q[15:0];
                    A_WBASE: r_wbase<=wdata_q;
                    default: ;
                endcase
                s_axil_bvalid<=1'b1; aw_hs<=1'b0; w_hs<=1'b0;
            end else if (s_axil_bvalid && s_axil_bready) s_axil_bvalid<=1'b0;
        end
    end

    logic [C_AXIL_AW-1:0] araddr_q;
    always_ff @(posedge clk) begin
        if (rst) begin
            s_axil_arready<=1'b0; s_axil_rvalid<=1'b0; s_axil_rdata<=32'd0;
        end else begin
            if (s_axil_arvalid && !s_axil_arready && !s_axil_rvalid) begin
                s_axil_arready<=1'b1; araddr_q<=s_axil_araddr;
            end else s_axil_arready<=1'b0;

            if (s_axil_arready) begin
                s_axil_rvalid<=1'b1;
                if (araddr_q[12]) s_axil_rdata <= {16'd0, logitmem[araddr_q[11:2]]}; // 0x1000+ logits
                else case (araddr_q)
                    A_STAT:  s_axil_rdata <= {30'd0, busy, done_latched};
                    A_TOK:   s_axil_rdata <= {16'd0, r_token};
                    A_POS:   s_axil_rdata <= {16'd0, r_pos};
                    A_WBASE: s_axil_rdata <= r_wbase;
                    default: s_axil_rdata <= 32'd0;
                endcase
            end else if (s_axil_rvalid && s_axil_rready) s_axil_rvalid<=1'b0;
        end
    end
    assign s_axil_rresp = 2'b00;
endmodule
