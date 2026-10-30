@echo off
rem Compile, elaborate and run one testbench with Vivado xsim.
rem Usage (from the project root):  sim\run.bat tb_mac
rem The source list for each testbench is in sim\<tb>.f
if "%1"=="" (
  echo usage: sim\run.bat ^<testbench^>
  exit /b 1
)
set VIVADO_BIN=C:\Xilinx\2025.1\Vivado\bin
if not exist build mkdir build
rem $readmemh paths are relative to where xsim runs, so put the ROM images there
if exist data\*.mem copy /y data\*.mem build >nul
cd build
call %VIVADO_BIN%\xvlog.bat -sv -f ..\sim\%1.f || (cd .. & exit /b 1)
call %VIVADO_BIN%\xelab.bat %1 -debug typical -s %1_sim || (cd .. & exit /b 1)
call %VIVADO_BIN%\xsim.bat %1_sim -runall
cd ..
