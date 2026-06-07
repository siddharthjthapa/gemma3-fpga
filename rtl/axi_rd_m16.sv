`timescale 1ns/1ps
//============================================================================
// axi_rd_m16 - 64-bit AXI4 read master, fp16 payload. Two output taps:
//   * WIDE   tap (w_data[63:0] = 4 fp16, 1 beat/cycle) - matmul weight quads
//   * NARROW tap (s_data[15:0], 4 fp16/beat)           - small reads
// `nwords` is the fp16 (16-bit) word count, MUST be a multiple of 4 (the caller
// rounds small reads up to 4 and ignores the extras). beats = nwords/4.
// `base` is a byte address (= halfword_offset * 2).
//============================================================================
module axi_rd_m16 #(
    parameter int AW = 32
) (
    input  logic            clk,
    input  logic            rstn,
    input  logic            start,
    input  logic            wide,        // 1: matmul (64-bit tap), 0: narrow
    input  logic [AW-1:0]   base,        // byte address
    input  logic [31:0]     nwords,      // fp16 words to read (multiple of 4)
    output logic            done,
    // AXI read address channel
    output logic [AW-1:0]   araddr,
    output logic [7:0]      arlen,
    output logic [2:0]      arsize,
    output logic [1:0]      arburst,
    output logic            arvalid,
    input  logic            arready,
    // AXI read data channel (64-bit)
    input  logic [63:0]     rdata,
    input  logic            rvalid,
    output logic            rready,
    input  logic            rlast,
    // narrow 16-bit word stream
    output logic [15:0]     s_data,
    output logic            s_valid,
    input  logic            s_ready,
    // wide 64-bit beat stream (4 fp16)
    output logic [63:0]     w_data,
    output logic            w_valid,
    input  logic            w_ready
);
    assign arsize  = 3'd3;      // 8 bytes/beat
    assign arburst = 2'b01;     // INCR

    typedef enum logic [1:0] {IDLE, ADDR, DATA, FIN} state_t;
    state_t st;

    logic [AW-1:0] addr;
    logic [31:0]   rem_beats;       // remaining beats this transfer
    logic [8:0]    blen;            // beats this burst (1..256)
    logic [1:0]    sub;             // narrow serializer: which fp16 in the beat
    logic          wide_q;

    // burst length (beats): min(remaining, 256, beats-to-next-4KB-boundary).
    wire [9:0]  cap256   = (rem_beats > 32'd256) ? 10'd256 : rem_beats[9:0];
    wire [12:0] to_4k    = 13'd4096 - {1'b0, addr[11:0]};
    wire [9:0]  beats_4k = to_4k[12:3];
    wire [9:0]  this_burst = (cap256 < beats_4k) ? cap256 : beats_4k;

    // data taps
    assign w_data  = rdata;
    assign w_valid = (st==DATA) &&  wide_q && rvalid;
    assign s_data  = rdata[16*sub +: 16];
    assign s_valid = (st==DATA) && !wide_q && rvalid;
    // consume a beat: wide -> every accepted beat; narrow -> after the 4th fp16
    wire beat_done = wide_q ? (w_valid && w_ready)
                            : (s_valid && s_ready && (sub==2'd3));
    assign rready  = beat_done;

    always_ff @(posedge clk) begin
        if (!rstn) begin
            st <= IDLE; arvalid <= 1'b0; done <= 1'b0; sub <= 2'd0;
        end else begin
            done <= 1'b0;
            case (st)
                IDLE: if (start) begin
                    addr <= base; rem_beats <= nwords >> 2; wide_q <= wide;
                    sub <= 2'd0; st <= ADDR;
                end
                ADDR: begin
                    araddr  <= addr;
                    arlen   <= this_burst[7:0] - 8'd1;
                    blen    <= this_burst[8:0];
                    arvalid <= 1'b1;
                    if (arvalid && arready) begin arvalid <= 1'b0; st <= DATA; end
                end
                DATA: begin
                    if (!wide_q && s_valid && s_ready) sub <= sub + 2'd1;
                    if (beat_done && rlast) begin
                        addr      <= addr + (blen << 3);   // 8 bytes/beat
                        rem_beats <= rem_beats - blen;
                        st <= (rem_beats - blen == 32'd0) ? FIN : ADDR;
                    end
                end
                FIN: begin done <= 1'b1; st <= IDLE; end
                default: st <= IDLE;
            endcase
        end
    end
endmodule
