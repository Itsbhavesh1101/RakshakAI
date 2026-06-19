from __future__ import annotations

import math
import wave
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from config import system_config


class AudioIntelligenceService:
    """Lightweight audio-event analyzer for gunshot, scream, and glass-break candidates."""

    def analyze_wav(self, audio_path: Path) -> Dict[str, Any]:
        if not system_config.audio_intelligence_enabled:
            return {"enabled": False, "events": [], "summary": "Audio intelligence disabled."}

        try:
            samples, sample_rate = self._read_wav(audio_path)
        except Exception as exc:
            return {"enabled": True, "events": [], "summary": f"Audio analysis failed: {exc}"}

        if samples.size == 0:
            return {"enabled": True, "events": [], "summary": "Audio file contains no samples."}

        features = self._features(samples, sample_rate)
        events = self._classify(features)
        return {
            "enabled": True,
            "sample_rate": sample_rate,
            "duration_seconds": round(samples.size / max(sample_rate, 1), 3),
            "features": features,
            "events": events,
            "summary": self._summary(events),
        }

    @staticmethod
    def _read_wav(audio_path: Path) -> tuple[np.ndarray, int]:
        with wave.open(str(audio_path), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            frames = handle.readframes(handle.getnframes())

        if sample_width == 1:
            samples = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0
            scale = 128.0
        elif sample_width == 2:
            samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
            scale = 32768.0
        elif sample_width == 4:
            samples = np.frombuffer(frames, dtype=np.int32).astype(np.float32)
            scale = 2147483648.0
        else:
            raise ValueError(f"Unsupported WAV sample width: {sample_width}")

        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
        return samples / scale, sample_rate

    @staticmethod
    def _features(samples: np.ndarray, sample_rate: int) -> Dict[str, float]:
        rms = float(np.sqrt(np.mean(samples * samples)))
        peak = float(np.max(np.abs(samples)))
        zero_crossings = float(np.mean(np.abs(np.diff(np.signbit(samples)))))

        window_size = max(256, min(4096, int(sample_rate * 0.04)))
        windows = [
            samples[index:index + window_size]
            for index in range(0, max(1, samples.size - window_size), window_size)
        ] or [samples]
        energies = np.asarray([float(np.sqrt(np.mean(window * window))) for window in windows if window.size], dtype=np.float32)
        burst_ratio = float(np.max(energies) / (np.mean(energies) + 1e-6)) if energies.size else 0.0

        spectrum = np.abs(np.fft.rfft(samples[: min(samples.size, sample_rate * 4)]))
        freqs = np.fft.rfftfreq(min(samples.size, sample_rate * 4), d=1.0 / sample_rate)
        spectral_centroid = float(np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-6)) if spectrum.size else 0.0
        high_band = float(np.sum(spectrum[freqs > 3000]) / (np.sum(spectrum) + 1e-6)) if spectrum.size else 0.0

        return {
            "rms": round(rms, 5),
            "peak": round(peak, 5),
            "zero_crossing_rate": round(zero_crossings, 5),
            "burst_ratio": round(burst_ratio, 3),
            "spectral_centroid_hz": round(spectral_centroid, 2),
            "high_frequency_ratio": round(high_band, 5),
        }

    @staticmethod
    def _classify(features: Dict[str, float]) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        peak = features["peak"]
        burst = features["burst_ratio"]
        centroid = features["spectral_centroid_hz"]
        high_band = features["high_frequency_ratio"]
        zcr = features["zero_crossing_rate"]

        if peak >= system_config.audio_gunshot_peak_threshold and burst >= 4.0:
            events.append({
                "type": "gunshot_candidate",
                "confidence": min(0.99, round(0.55 + peak * 0.25 + min(burst, 10.0) * 0.03, 3)),
                "evidence": {"peak": peak, "burst_ratio": burst, "rule": "impulsive_loud_burst"},
            })
        if centroid >= system_config.audio_scream_centroid_threshold and zcr >= 0.08:
            events.append({
                "type": "scream_candidate",
                "confidence": min(0.99, round(0.45 + min(centroid / 8000.0, 0.4) + min(zcr, 0.2), 3)),
                "evidence": {"spectral_centroid_hz": centroid, "zero_crossing_rate": zcr, "rule": "high_centroid_voice_like_event"},
            })
        if high_band >= system_config.audio_glass_high_freq_ratio and burst >= 2.0:
            events.append({
                "type": "glass_break_candidate",
                "confidence": min(0.99, round(0.45 + high_band * 1.2 + min(burst, 8.0) * 0.04, 3)),
                "evidence": {"high_frequency_ratio": high_band, "burst_ratio": burst, "rule": "high_frequency_sharp_event"},
            })
        return sorted(events, key=lambda item: item["confidence"], reverse=True)

    @staticmethod
    def _summary(events: List[Dict[str, Any]]) -> str:
        if not events:
            return "No configured audio threat signature detected."
        top = events[0]
        return f"{top['type'].replace('_', ' ')} detected with confidence {top['confidence']:.2f}."
