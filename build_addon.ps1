$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$dist = Join-Path $root "dist"
$staging = Join-Path $dist "package"
# Read the addon version from manifest.ini so we don't have to keep this
# script in sync when the version is bumped.
$manifestPath = Join-Path $root "manifest.ini"
$versionLine = Get-Content $manifestPath | Where-Object { $_ -match '^version\s*=' } | Select-Object -First 1
if ($null -eq $versionLine) {
	throw "Could not find 'version = ...' line in manifest.ini"
}
$version = ($versionLine -split '=', 2)[1].Trim()
if ($version -notmatch '^\d+(\.\d+)*$') {
	throw "manifest.ini version '$version' is not a plain dotted version (e.g. 0.6.0)"
}
$output = Join-Path $dist "groqVoiceDictation-$version.nvda-addon"
$zipOutput = Join-Path $dist "groqVoiceDictation-$version.zip"

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
# Compress-Archive intermittently fails on large vendored dependency trees with
# spurious "file is being used by another process" errors, often naming a
# different untouched NumPy source file on each attempt. ZipFile traverses the
# staging directory in one pass without that PowerShell.Archive failure mode.
# Add each entry explicitly so its ZIP name uses forward slashes; the ZIP spec
# and NVDA's package loader expect '/' even when the build runs on Windows.
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive = [System.IO.Compression.ZipFile]::Open(
	$zipOutput,
	[System.IO.Compression.ZipArchiveMode]::Create
)
try {
	Get-ChildItem -LiteralPath $staging -Recurse -File | ForEach-Object {
		$entryName = $_.FullName.Substring($staging.Length + 1).Replace('\', '/')
		[System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
			$archive,
			$_.FullName,
			$entryName,
			[System.IO.Compression.CompressionLevel]::Optimal
		) | Out-Null
	}
}
finally {
	$archive.Dispose()
}
Move-Item -Force $zipOutput $output
Write-Host "Built $output"
