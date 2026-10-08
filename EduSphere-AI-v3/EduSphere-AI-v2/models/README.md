# Passive face liveness model (MiniFASNet-V2)

EduSphere's current adapter uses **MiniFASNet-V2 ONNX** and checks each of three
consecutive camera frames independently. The lowest live probability must meet
the configured threshold before the identity can be marked present.

This is a per-frame anti-spoofing model with three-frame consensus. It is **not**
a temporal neural network and cannot guarantee detection of every printed,
replayed, mask, deepfake, or camera-injection attack. Test it with the actual
mobile camera, classroom distance, lighting, and consented test subjects before
real attendance use.

## Download

Download the ONNX file from the model repository:

<https://huggingface.co/garciafido/minifasnet-v2-anti-spoofing-onnx/resolve/main/minifasnet_v2.onnx>

Save it in this folder with this exact filename:

`face_anti_spoof.onnx`

Model card and preprocessing: <https://huggingface.co/garciafido/minifasnet-v2-anti-spoofing-onnx>

- License: Apache-2.0 (the model repository includes the license file).
- Expected input: `[1, 3, 80, 80]`, float32, **BGR**, values in `[0, 1]`.
- Output classes: `[live, print-attack, replay-attack]`; live class index is `0`.
- Face crop: square crop centered on the detected face, scaled by `2.7`, resized to `80×80`.
- ONNX SHA-256: `d7b3cd9ba8a7ceb13baa8c4720902e27ca3112eff52f926c08804af6b6eecc7b`.
- File size: 1,744,116 bytes.

## Windows download helper

From the `ai-service` folder, run:

`powershell -ExecutionPolicy Bypass -File .\download_liveness_model.ps1`

The script downloads the published file and verifies its SHA-256 before keeping it.

Until the model is installed and `/health` reports
`passive_liveness_model_loaded: true`, AI attendance stays blocked. Manual
attendance remains available.
