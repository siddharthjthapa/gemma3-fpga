`timescale 1ns/1ps
//============================================================================
// tb_pmac16 - drive pmac16 with K fp16 dot-products of length L and compare
//   the reordered fp16 result against the numpy 8-bucket reference (exact).
//============================================================================
module tb_pmac16;
    localparam int K=64, L=80;
    logic clk=0, rst=1; always #5 clk=~clk;

    logic [15:0] X [0:K*L-1], W [0:K*L-1], R [0:K-1];
    logic start, feed, last; logic [15:0] x, w; wire done; wire [15:0] result;

    pmac16 DUT (.clk(clk), .rst(rst), .start(start), .feed(feed),
                .x(x), .w(w), .last(last), .done(done), .result(result));

    function automatic int ulp(input logic [15:0] xx, input logic [15:0] r);
        int xi, ri;
        xi = xx[15] ? (32'h8000 - {16'd0,xx[14:0]}) : (32'h8000 + {16'd0,xx[14:0]});
        ri = r[15]  ? (32'h8000 - {16'd0,r[14:0]})  : (32'h8000 + {16'd0,r[14:0]});
        return (xi>ri)?(xi-ri):(ri-xi);
    endfunction

    integer k, j, exact, maxulp, d;
    initial begin
        $readmemh("pm_x.hex", X); $readmemh("pm_w.hex", W); $readmemh("pm_ref.hex", R);
        start=0; feed=0; last=0; x=0; w=0;
        repeat(4) @(posedge clk); rst=0; @(posedge clk);
        exact=0; maxulp=0;
        for (k=0; k<K; k=k+1) begin
            @(posedge clk); start<=1;
            @(posedge clk); start<=0;            // ACC reached after 1 cyc
            for (j=0; j<L; j=j+1) begin
                feed<=1; x<=X[k*L+j]; w<=W[k*L+j]; last<=(j==L-1);
                @(posedge clk);
            end
            feed<=0; last<=0;
            wait(done); #1;
            d=ulp(result, R[k]);
            if (d==0) exact=exact+1; else if (d>maxulp) maxulp=d;
            @(posedge clk);
        end
        $display("PMAC16: exact %0d/%0d  max_ulp=%0d", exact, K, maxulp);
        if (maxulp<=1) $display("RESULT: PASS"); else $display("RESULT: FAIL");
        $finish;
    end
    initial begin #2000000 $display("RESULT: TIMEOUT"); $finish; end
endmodule
