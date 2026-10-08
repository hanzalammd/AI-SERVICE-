$ErrorActionPreference = 'Stop'
$modelUrl = 'https://huggingface.co/garciafido/minifasnet-v2-anti-spoofing-onnx/resolve/main/minifasnet_v2.onnx'
$expectedHash = 'd7b3cd9ba8a7ceb13baa8c4720902e27ca3112eff52f926c08804af6b6eecc7b'
$modelDir = Join-Path $PSScriptRoot 'models'
$modelPath = Join-Path $modelDir 'face_anti_spoof.onnx'
New-Item -ItemType Directory -Force -Path $modelDir | Out-Null
Write-Host 'Downloading MiniFASNet-V2 ONNX model...'
Invoke-WebRequest -Uri $modelUrl -OutFile $modelPath
$actualHash = (Get-FileHash -Path $modelPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualHash -ne $expectedHash) {
    Remove-Item -Force $modelPath
    throw "Model checksum mismatch. Expected $expectedHash but received $actualHash. The downloaded file was removed."
}
Write-Host 'Model downloaded and SHA-256 verified.' -ForegroundColor Green
Write-Host "Saved to: $modelPath"
