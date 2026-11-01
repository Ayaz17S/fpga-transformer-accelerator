@echo off
rem Run every testbench and print one PASS/FAIL line each.
rem Usage (from the project root):  sim\regress.bat
for %%t in (tb_mac tb_matvec tb_embed tb_rmsnorm tb_softmax tb_attention tb_gpt_core tb_generator tb_axi_gpt) do (
  echo == %%t
  call sim\run.bat %%t 2>&1 | findstr /C:"PASS" /C:"FAIL" /C:"ERROR" /C:"Fatal"
)
