<#
  run.ps1 - Build and/or flash the Gemma3-600K PL accelerator on the Z-turn Lite.

  Usage (from this folder):
    .\run.ps1            Flash existing bitstream + ELF + weights over JTAG (default)
    .\run.ps1 -Build     Rebuild assets (model.bin + vocab.h) + app ELF, then flash
    .\run.ps1 -All       Rebuild bitstream (Vivado) + assets + ELF, then flash
    .\run.ps1 -NoFlash   Build only, do not flash

  Open the J24 UART (PS UART1) in PuTTY @ 115200 8N1 to watch the generated story.
#>
[CmdletBinding()]
param([switch]$Build, [switch]$All, [switch]$NoFlash)
$ErrorActionPreference = 'Stop'

$Proj   = $PSScriptRoot
$Script = Join-Path (Split-Path $Proj -Parent) "script"
$Vivado = "C:\Xilinx\2025.1\Vivado\bin\vivado.bat"
$Vitis  = "C:\Xilinx\2025.1\Vitis\bin\vitis.bat"
$Xsdb   = "C:\Xilinx\2025.1\Vivado\bin\xsdb.bat"
$Python = "python"

$Xsa   = Join-Path $Proj "gemma_soc.xsa"
$Bit   = Join-Path $Proj "gemma_soc.bit"
$Model = Join-Path $Proj "model.bin"
$Elf   = Join-Path $Proj "ws\gemma_app\build\gemma_app.elf"

function Step($m) { Write-Host "`n==== $m ====" -ForegroundColor Cyan }
function Die($m)  { Write-Host "ERROR: $m" -ForegroundColor Red; exit 1 }

# --- 1. hardware (Vivado -> bitstream + XSA) --------------------------------
if ($All -or -not (Test-Path $Bit)) {
    if (-not (Test-Path $Vivado)) { Die "Vivado not found at $Vivado" }
    Step "Building bitstream (Vivado)  [~15-20 min]"
    & $Vivado -mode batch -notrace -source (Join-Path $Proj "build_gemma.tcl")
    if (-not (Test-Path $Bit)) { Die "bitstream not produced - check build logs" }
}

# --- 2. assets: model.bin (fp16 blob) + vocab.h ----------------------------
if ($All -or $Build -or -not (Test-Path $Model)) {
    Step "Generating model.bin (gen_blob.py) + vocab.h (gen_assets.py)"
    & $Python (Join-Path $Script "gen_blob.py")      # writes soc\model.bin
    & $Python (Join-Path $Proj "gen_assets.py")      # writes sw\vocab.h
    if (-not (Test-Path $Model)) { Die "model.bin not produced" }
}

# --- 3. application (Vitis -> ELF) -----------------------------------------
if ($All -or $Build -or -not (Test-Path $Elf)) {
    if (-not (Test-Path $Vitis)) { Die "Vitis not found at $Vitis" }
    Step "Building application (Vitis -> ELF)  [a few minutes]"
    & $Vitis -s (Join-Path $Proj "build_sw.py")
    if (-not (Test-Path $Elf)) { Die "ELF not produced - check build_sw log" }
}

if ($NoFlash) { Step "Build complete (no flash requested)"; exit 0 }

# --- 4. flash + run --------------------------------------------------------
$Ps7 = (Get-ChildItem -Path (Join-Path $Proj "ws") -Recurse -Filter ps7_init.tcl -ErrorAction SilentlyContinue |
        Select-Object -First 1).FullName
if (-not $Ps7) { Die "ps7_init.tcl not found under ws\" }

Step "Flashing (program PL, load weights to DDR, run app)"
& $Xsdb (Join-Path $Proj "flash_gemma.tcl") $Ps7 $Bit $Model $Elf
Write-Host "`nOpen J24 UART @115200 8N1 to read the story." -ForegroundColor Green
