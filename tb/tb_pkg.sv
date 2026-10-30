// tb_pkg.sv - reference functions shared by all testbenches.

`timescale 1ns/1ps

package tb_pkg;

  // Same rule as the contract's Python requant(): round half up,
  // arithmetic shift right, saturate to 16 bits.
  function automatic longint requant_ref(longint acc, int sh);
    if (sh > 0) acc = (acc + (longint'(1) << (sh - 1))) >>> sh;
    if (acc >  32767) acc =  32767;
    if (acc < -32768) acc = -32768;
    return acc;
  endfunction

endpackage
