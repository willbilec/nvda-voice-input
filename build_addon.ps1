$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$dist = Join-Path $root "dist"
$staging = Join-Path $dist "package"
$output = Join-Path $dist "groqVoiceDictation-0.5.0.nvda-addon"
$zipOutput = Join-Path $dist "groqVoiceDictation-0.5.0.zip"

if (Test-Path $staging) {
	Remove-Item -Recurse -Force $staging
}
New-Item -ItemType Directory -Force -Path $staging | Out-Null

Copy-Item (Join-Path $root "manifest.ini") $staging
Copy-Item (Join-Path $root "doc") $staging -Recurse
Copy-Item (Join-Path $root "globalPlugins") $staging -Recurse

Get-ChildItem $staging -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
Get-ChildItem $staging -Recurse -File -Include "*.pyc","*.pyo" | Remove-Item -Force

if (Test-Path $output) {
	Remove-Item -Force $output
}
if (Test-Path $zipOutput) {
	Remove-Item -Force $zipOutput
}
Compress-Archive -Path (Join-Path $staging "*") -DestinationPath $zipOutput
Move-Item -Force $zipOutput $output
Write-Host "Built $output"
