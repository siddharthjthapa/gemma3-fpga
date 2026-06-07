<#
  sim.ps1 - compile + run an xsim simulation for the gemma_pl project.

  Usage (from anywhere):
    .\script\sim.ps1 -Top tb_fp16 -Files "fp16_mul_p.sv,fp16_add_p.sv" -Tb "tb_fp16.sv"

  -Top    : top testbench module name
  -Files  : comma-separated RTL file names (in rtl/)
  -Tb     : comma-separated testbench file names (in sim/)
  Runs from sim/ so $readmemh relative paths and xsim.dir land there.
#>
[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)][string]$Top,
  [string]$Files = "",
  [string]$Tb = ""
)
$ErrorActionPreference = 'Stop'
$BIN = "C:\Xilinx\2025.1\Vivado\bin"
$ROOT = Split-Path $PSScriptRoot -Parent
$RTL  = Join-Path $ROOT "rtl"
$SIM  = Join-Path $ROOT "sim"

$src = @()
foreach ($f in ($Files -split ',')) { if ($f.Trim()) { $src += (Join-Path $RTL $f.Trim()) } }
foreach ($f in ($Tb    -split ',')) { if ($f.Trim()) { $src += (Join-Path $SIM $f.Trim()) } }

Push-Location $SIM
try {
  Write-Host "==== xvlog ====" -ForegroundColor Cyan
  & "$BIN\xvlog.bat" --sv $src
  if ($LASTEXITCODE -ne 0) { throw "xvlog failed" }
  Write-Host "==== xelab $Top ====" -ForegroundColor Cyan
  & "$BIN\xelab.bat" $Top -s "${Top}_snap" --timescale 1ns/1ps
  if ($LASTEXITCODE -ne 0) { throw "xelab failed" }
  Write-Host "==== xsim $Top ====" -ForegroundColor Cyan
  & "$BIN\xsim.bat" "${Top}_snap" -runall
  if ($LASTEXITCODE -ne 0) { throw "xsim failed" }
} finally {
  Pop-Location
}
