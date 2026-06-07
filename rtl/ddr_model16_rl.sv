`timescale 1ns/1ps
//============================================================================
// ddr_model16_rl - REALISTIC fp16 AXI4 read slave (vs the idealized
//   ddr_model16). Models what the real PS7 HP0 port does and the idealized
//   model does NOT: AR-accept latency, AR->first-data latency, and rvalid
//   GAPS mid-burst (slave not always ready to hand over the next beat).
//   Used to flush out axi_rd_m16 handshake bugs that only bite on hardware.
//   64-bit beat = 4 fp16 {mem[h+3..h]}, byte addr -> halfword index = araddr>>1.
//============================================================================
module ddr_model16_rl #(
    parameter int AW    = 32,
    parameter int WORDS = 4096,
    parameter int AR_LAT = 3,      // cycles arvalid high before arready
    parameter int R_LAT  = 5       // cycles after AR accept before first beat
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

    typedef enum logic [1:0] {IDLE, ARWAIT, RWAIT, RESP} state_t;
    state_t st;
    logic [AW-1:0] hbase;
    logic [8:0]    cnt, len;
    logic [7:0]    dly;
    logic [15:0]   lfsr;
    logic          rv;             // internal rvalid (with gaps)

    assign rresp = 2'b00;
    assign arready = (st == ARWAIT) && (dly == 0);
    assign rdata = {mem[hbase + 4*cnt + 3], mem[hbase + 4*cnt + 2],
                    mem[hbase + 4*cnt + 1], mem[hbase + 4*cnt]};
    assign rvalid = (st == RESP) && rv;
    assign rlast  = (st == RESP) && rv && (cnt == len);

    // pseudo-random gap: when entering a beat, sometimes stall rvalid 1-2 cycles
    wire gap = (lfsr[2:0] == 3'b000);     // ~1/8 beats get a gap

    always_ff @(posedge clk) begin
        if (!rstn) begin
            st <= IDLE; cnt <= 0; dly <= 0; lfsr <= 16'hACE1; rv <= 1'b0;
        end else begin
            lfsr <= {lfsr[14:0], lfsr[15]^lfsr[13]^lfsr[12]^lfsr[10]};
            case (st)
                IDLE: if (arvalid) begin
                    // REAL HP aligns the start address DOWN to the ARSIZE (8-byte)
                    // boundary. araddr>>1 = halfword index; &~3 = 4-halfword align.
                    hbase <= (araddr >> 1) & ~16'd3; len <= {1'b0, arlen};
                    dly <= AR_LAT[7:0]; st <= ARWAIT;
                end
                ARWAIT: begin
                    if (dly != 0) dly <= dly - 8'd1;
                    else if (arvalid && arready) begin
                        cnt <= 0; dly <= R_LAT[7:0]; rv <= 1'b0; st <= RWAIT;
                    end
                end
                RWAIT: if (dly != 0) dly <= dly - 8'd1; else begin rv <= 1'b1; st <= RESP; end
                RESP: begin
                    if (rv && rready) begin
                        if (cnt == len) begin st <= IDLE; rv <= 1'b0; end
                        else begin
                            cnt <= cnt + 9'd1;
                            rv <= ~gap;            // maybe drop rvalid for the next beat
                        end
                    end else if (!rv) begin
                        rv <= 1'b1;                // end the gap
                    end
                end
            endcase
        end
    end
endmodule
