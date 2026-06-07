`timescale 1ns/1ps
//============================================================================
// ddr_model16 - minimal AXI4 64-bit read-only slave backed by a 16-bit word
//   memory (loaded from model.hex, one fp16 per line). Each 64-bit beat =
//   {mem[h+3], mem[h+2], mem[h+1], mem[h]} (4 little-endian fp16). Combinational
//   data/valid/last. byte address araddr -> halfword index = araddr>>1.
//============================================================================
module ddr_model16 #(
    parameter int AW   = 32,
    parameter int WORDS = 4096
) (
    input  logic          clk,
    input  logic          rstn,
    input  logic [AW-1:0] araddr,
    input  logic [7:0]    arlen,
    input  logic [2:0]    arsize,
    input  logic [1:0]    arburst,
    input  logic          arvalid,
    output logic          arready,
    output logic [63:0]   rdata,
    output logic          rvalid,
    output logic          rlast,
    output logic [1:0]    rresp,
    input  logic          rready
);
    logic [15:0] mem [0:WORDS-1];

    typedef enum logic [0:0] {IDLE, RESP} state_t;
    state_t st;
    logic [AW-1:0] hbase;       // halfword base (= araddr>>1, 4-aligned)
    logic [8:0]    cnt, len;

    assign rresp   = 2'b00;
    assign arready = (st == IDLE);
    assign rvalid  = (st == RESP);
    assign rdata   = {mem[hbase + 4*cnt + 3], mem[hbase + 4*cnt + 2],
                      mem[hbase + 4*cnt + 1], mem[hbase + 4*cnt]};
    assign rlast   = (st == RESP) && (cnt == len);

    always_ff @(posedge clk) begin
        if (!rstn) begin
            st <= IDLE; cnt <= 9'd0;
        end else begin
            case (st)
                IDLE: if (arvalid && arready) begin
                    hbase <= araddr >> 1;
                    len   <= {1'b0, arlen};
                    cnt   <= 9'd0;
                    st    <= RESP;
                end
                RESP: if (rvalid && rready) begin
                    if (cnt == len) st <= IDLE;
                    else            cnt <= cnt + 9'd1;
                end
            endcase
        end
    end
endmodule
