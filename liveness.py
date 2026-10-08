"""Passive per-frame and temporal face presentation-attack detection with 3-frame consensus.

Supports MiniFASNet-V2 ONNX per-frame 3-frame consensus and concatenated temporal
multi-frame models. Wires LIVENESS_MODEL_PATH, LIVENESS_LIVE_CLASS_INDEX,
LIVENESS_INPUT_SIZE, LIVENESS_INPUT_SCALE, and LIVENESS_THRESHOLD.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


class PassiveLiveness:
    def __init__(self) -> None:
        self.path = os.getenv("LIVENESS_MODEL_PATH", "./models/face_anti_spoof.onnx").strip()
        self.threshold = float(os.getenv("LIVENESS_THRESHOLD", "0.90"))
        self.live_index = int(os.getenv("LIVENESS_LIVE_CLASS_INDEX", "0"))
        self.input_size = int(os.getenv("LIVENESS_INPUT_SIZE", "80"))
        self.crop_scale = float(os.getenv("LIVENESS_INPUT_SCALE", "2.7"))
        self.session = None
        self.input_name = None
        self.output_name = None
        self.is_temporal_concat = False

        raw_path = self.path
        model_file = Path(raw_path)
        if not model_file.is_file():
            # Check relative to ai-service directory
            cand = Path(__file__).resolve().parent / raw_path
            if cand.is_file():
                model_file = cand
                self.path = str(cand)
            else:
                cand2 = Path(__file__).resolve().parent / "models" / "face_anti_spoof.onnx"
                if cand2.is_file():
                    model_file = cand2
                    self.path = str(cand2)

        if model_file.is_file():
            session = ort.InferenceSession(str(model_file), providers=["CPUExecutionProvider"])
            model_input = session.get_inputs()[0]
            shape = model_input.shape
            if len(shape) != 4:
                raise ValueError(f"Anti-spoofing model must accept a 4D image input [N,C,H,W], got shape: {shape}.")

            channels = shape[1]
            if isinstance(channels, int):
                if channels == 9:
                    self.is_temporal_concat = True
                elif channels not in (1, 3):
                    raise ValueError(f"Anti-spoofing input channels must be 3 (BGR) or 9 (temporal 3-frame), got {channels}.")

            if isinstance(shape[2], int) and shape[2] > 0:
                self.input_size = shape[2]
            if isinstance(shape[3], int) and shape[3] > 0 and shape[3] != self.input_size:
                raise ValueError(f"Model height ({shape[2]}) and width ({shape[3]}) must match.")

            outputs = session.get_outputs()
            if not outputs:
                raise ValueError("Anti-spoofing ONNX model has no output tensor.")
            self.session = session
            self.input_name = model_input.name
            self.output_name = outputs[0].name

    @property
    def ready(self) -> bool:
        return self.session is not None

    def _crop(self, image: np.ndarray, bbox) -> np.ndarray:
        height, width = image.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in bbox]
        face_w, face_h = max(1.0, x2 - x1), max(1.0, y2 - y1)
        side = max(face_w, face_h) * self.crop_scale
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        left = max(0, int(round(cx - side / 2.0)))
        top = max(0, int(round(cy - side / 2.0)))
        right = min(width, int(round(cx + side / 2.0)))
        bottom = min(height, int(round(cy + side / 2.0)))
        crop = image[top:bottom, left:right]
        if crop.size == 0 or min(crop.shape[:2]) < 16:
            raise ValueError("Face crop is too small for passive liveness analysis.")
        crop = cv2.resize(crop, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        tensor = crop.astype(np.float32) / 255.0
        return np.transpose(tensor, (2, 0, 1)).astype(np.float32)

    def _normalize_live_prob(self, output: np.ndarray) -> float:
        vec = output.reshape(-1)
        if vec.size <= self.live_index:
            raise RuntimeError(f"Model output size ({vec.size}) is smaller than live class index ({self.live_index}).")
        if np.any(vec < 0) or not np.isclose(float(np.sum(vec)), 1.0, atol=1e-3):
            shifted = vec - np.max(vec)
            exp = np.exp(shifted)
            vec = exp / max(float(np.sum(exp)), 1e-12)
        return float(vec[self.live_index])

    def score_track(self, observations: list[tuple[np.ndarray, object]]) -> float:
        if not self.ready:
            raise RuntimeError("Passive anti-spoofing model is not configured.")
        if len(observations) != 3:
            raise ValueError("Exactly three matched camera frames are required for liveness consensus.")

        # Static replay detector: if all 3 frames are bit-identical, reject immediately.
        # Live camera captures always have natural sensor noise / micro-movement across consecutive frames.
        f0 = observations[0][0]
        f1 = observations[1][0]
        f2 = observations[2][0]
        if f0.shape == f1.shape == f2.shape:
            diff01 = float(np.mean(np.abs(f0.astype(np.float32) - f1.astype(np.float32))))
            diff12 = float(np.mean(np.abs(f1.astype(np.float32) - f2.astype(np.float32))))
            if diff01 < 0.05 and diff12 < 0.05:
                # Replay attack with identical / frozen static image
                return 0.0

        if self.is_temporal_concat:
            # 9-channel temporal model: concat 3 crops along channel dimension
            crops = [self._crop(img, bbox) for img, bbox in observations[:3]]
            tensor = np.concatenate(crops, axis=0)[None, ...]  # shape: [1, 9, H, W]
            raw_out = self.session.run([self.output_name], {self.input_name: tensor})[0]
            return self._normalize_live_prob(np.asarray(raw_out, dtype=np.float32))
        else:
            # Batch evaluate all 3 frames together: shape [3, 3, H, W]
            crops = np.stack([self._crop(img, bbox) for img, bbox in observations[:3]], axis=0)
            raw_out = self.session.run([self.output_name], {self.input_name: crops})[0]
            scores = [self._normalize_live_prob(np.asarray(row, dtype=np.float32)) for row in raw_out]
            # 3-frame consensus: minimum probability among the 3 frames
            return float(min(scores))
