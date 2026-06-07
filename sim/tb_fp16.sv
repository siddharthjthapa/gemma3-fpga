`timescale 1ns/1ps
//============================================================================
// tb_fp16 - unit test for fp16_mul_p (lat 3) and fp16_add_p (lat 5) against
//   numpy fp16 reference vectors. Reports exact-match count and max ULP error.
//============================================================================
module tb_fp16;
    localparam int N = 2000;
    logic clk=0; always #5 clk=~clk;

    logic [15:0] A [0:N-1], B [0:N-1], MR [0:N-1], AR [0:N-1];
    logic [15:0] a, b; wire [15:0] ym, ya;

    fp16_mul_p UM (.clk(clk), .a(a), .b(b), .y(ym));
    fp16_add_p UA (.clk(clk), .a(a), .b(b), .y(ya));

    // unsigned-magnitude ULP distance between two fp16 bit patterns
    function automatic int ulp(input logic [15:0] x, input logic [15:0] r);
        int xi, ri;
        xi = x[15] ? (32'h8000 - {16'd0,x[14:0]}) : (32'h8000 + {16'd0,x[14:0]});
        ri = r[15] ? (32'h8000 - {16'd0,r[14:0]}) : (32'h8000 + {16'd0,r[14:0]});
        return (xi>ri)?(xi-ri):(ri-xi);
    endfunction

    integer i, fed, mexact, aexact, mmaxulp, amaxulp, d;
    initial begin
        $readmemh("fp16_a.hex", A);
        $readmemh("fp16_b.hex", B);
        $readmemh("fp16_mul_ref.hex", MR);
        $readmemh("fp16_add_ref.hex", AR);
        a=0; b=0;
        mexact=0; aexact=0; mmaxulp=0; amaxulp=0;
        // feed one pair/cycle; results lag by the pipeline latency.
        for (i=0; i<N+8; i=i+1) begin
            @(posedge clk);
            if (i<N) begin a<=A[i]; b<=B[i]; end
            // mul result for index i-3 is valid this cycle (after the NBA)
            #1;
            if (i>=3 && i<N+3) begin
                d=ulp(ym, MR[i-3]);
                if (d==0) mexact=mexact+1; else if (d>mmaxulp) mmaxulp=d;
            end
            if (i>=5 && i<N+5) begin
                d=ulp(ya, AR[i-5]);
                if (d==0) aexact=aexact+1; else if (d>amaxulp) amaxulp=d;
            end
        end
        $display("MUL: exact %0d/%0d  max_ulp=%0d", mexact, N, mmaxulp);
        $display("ADD: exact %0d/%0d  max_ulp=%0d", aexact, N, amaxulp);
        if (mmaxulp<=1 && amaxulp<=1) $display("RESULT: PASS");
        else $display("RESULT: FAIL");
        $finish;
    end
endmodule
