`timescale 1ns/1ps
//============================================================================
// tb_trans16 - test fp16_exp_p and fp16_rsqrt_p against numpy fp16 references.
//   Reports worst relative error for each. PASS if exp<2% and rsqrt<0.5%.
//============================================================================
module tb_trans16;
    localparam int NE=64, NR=48;
    logic clk=0, rst=1; always #5 clk=~clk;

    logic [15:0] EX[0:NE-1], EXR[0:NE-1], RX[0:NR-1], RXR[0:NR-1];
    logic estart, rstart; logic [15:0] ex_x, rx_x; wire [15:0] ey, ry; wire edone, rdone;

    fp16_exp_p   UE (.clk(clk),.rst(rst),.start(estart),.x(ex_x),.y(ey),.done(edone));
    fp16_rsqrt_p UR (.clk(clk),.rst(rst),.start(rstart),.x(rx_x),.y(ry),.done(rdone));

    function automatic real f2r(input logic [15:0] b);
        real mant; int unsigned e; logic [9:0] m; logic s;
        s=b[15]; e=b[14:10]; m=b[9:0];
        if (e==0) return 0.0;
        mant=1.0; for (int k=0;k<10;k++) if (m[k]) mant+=2.0**(k-10);
        return (s?-1.0:1.0)*mant*(2.0**(int'(e)-15));
    endfunction
    function automatic real relerr(input logic [15:0] got, input logic [15:0] exp_b);
        real g,r; g=f2r(got); r=f2r(exp_b);
        if (r==0.0) return (g==0.0)?0.0:1.0;
        return ((g>r)?(g-r):(r-g))/((r<0)?-r:r);
    endfunction

    integer i; real e_worst, r_worst, e;
    initial begin
        $readmemh("exp_in.hex",EX); $readmemh("exp_ref.hex",EXR);
        $readmemh("rsq_in.hex",RX); $readmemh("rsq_ref.hex",RXR);
        estart=0; rstart=0; ex_x=0; rx_x=0;
        repeat(4) @(posedge clk); rst=0; @(posedge clk);
        e_worst=0;
        for (i=0;i<NE;i=i+1) begin
            ex_x<=EX[i]; @(posedge clk); estart<=1; @(posedge clk); estart<=0;
            wait(edone); #1; e=relerr(ey,EXR[i]); if (e>e_worst) e_worst=e; @(posedge clk);
        end
        r_worst=0;
        for (i=0;i<NR;i=i+1) begin
            rx_x<=RX[i]; @(posedge clk); rstart<=1; @(posedge clk); rstart<=0;
            wait(rdone); #1; e=relerr(ry,RXR[i]); if (e>r_worst) r_worst=e; @(posedge clk);
        end
        $display("EXP   worst rel err = %.4e", e_worst);
        $display("RSQRT worst rel err = %.4e", r_worst);
        if (e_worst<0.02 && r_worst<0.005) $display("RESULT: PASS");
        else $display("RESULT: FAIL");
        $finish;
    end
    initial begin #5000000 $display("RESULT: TIMEOUT"); $finish; end
endmodule
