"""Adaptive precision and quantization with dynamic switching.

Implements FP8, Int4, FP16, and FP32 precision modes with automatic
fallback for numerical stability.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class PrecisionMode(Enum):
    """Supported precision modes for computation."""
    FP32 = "fp32"
    FP16 = "fp16"
    FP8 = "fp8"
    INT8 = "int8"
    INT4 = "int4"
    BF16 = "bf16"


@dataclass
class PrecisionConfig:
    """Configuration for adaptive precision."""
    mode: PrecisionMode = PrecisionMode.FP32
    accumulator_mode: PrecisionMode = PrecisionMode.FP32
    tolerance: float = 1e-5
    auto_fallback: bool = True
    calibration_samples: int = 100


class AdaptivePrecisionManager:
    """Manages dynamic precision switching with numerical stability checks."""
    
    def __init__(self, config: Optional[PrecisionConfig] = None) -> None:
        self.config = config or PrecisionConfig()
        self._calibration_data: list[float] = []
        self._error_history: list[float] = []
        self._current_mode = self.config.mode
        
    def calibrate(self, reference_output: Any, quantized_output: Any) -> float:
        """Calibrate quantization based on reference output."""
        # Simulate error computation
        if hasattr(reference_output, "__len__"):
            ref_norm = sum(abs(x) for x in reference_output)
            quant_norm = sum(abs(x) for x in quantized_output)
            error = abs(ref_norm - quant_norm) / (ref_norm + 1e-10)
        else:
            error = abs(reference_output - quantized_output) / (abs(reference_output) + 1e-10)
            
        self._error_history.append(error)
        return error
        
    def should_fallback(self) -> bool:
        """Determine if we should fallback to higher precision."""
        if not self.config.auto_fallback:
            return False
            
        if len(self._error_history) < 10:
            return False
            
        # Check recent error history
        recent_errors = self._error_history[-10:]
        avg_error = sum(recent_errors) / len(recent_errors)
        
        return avg_error > self.config.tolerance
        
    def select_precision(self, op_type: str, input_range: tuple[float, float]) -> PrecisionMode:
        """Select appropriate precision for an operation."""
        min_val, max_val = input_range
        magnitude = max(abs(min_val), abs(max_val))
        
        # Unstable operations always use higher precision
        unstable_ops = {"div", "sqrt", "log", "exp", "softmax"}
        if op_type in unstable_ops:
            return PrecisionMode.FP16
            
        # Small magnitude values need higher precision
        if magnitude < 1e-3:
            return PrecisionMode.FP16
            
        # Check if lower precision is safe
        if self.should_fallback():
            return self._fallback_mode()
            
        return self._current_mode
        
    def _fallback_mode(self) -> PrecisionMode:
        """Get fallback precision mode."""
        fallback_chain = {
            PrecisionMode.INT4: PrecisionMode.INT8,
            PrecisionMode.INT8: PrecisionMode.FP8,
            PrecisionMode.FP8: PrecisionMode.FP16,
            PrecisionMode.FP16: PrecisionMode.FP32,
            PrecisionMode.BF16: PrecisionMode.FP32,
        }
        return fallback_chain.get(self._current_mode, PrecisionMode.FP32)
        
    def quantize(self, value: float, mode: PrecisionMode) -> float:
        """Quantize a value to specified precision."""
        if mode == PrecisionMode.FP32:
            return value
        elif mode == PrecisionMode.FP16:
            # Simulate FP16 quantization
            return round(value * 2048) / 2048
        elif mode == PrecisionMode.FP8:
            # Simulate FP8 quantization (E4M3 format)
            return round(value * 16) / 16
        elif mode == PrecisionMode.INT8:
            # Quantize to INT8 range [-128, 127]
            scale = 127.0 / max(abs(value), 1e-10)
            return round(value * scale) / scale
        elif mode == PrecisionMode.INT4:
            # Quantize to INT4 range [-8, 7]
            scale = 7.0 / max(abs(value), 1e-10)
            return round(value * scale) / scale
        else:
            return value
            
    def get_statistics(self) -> dict[str, Any]:
        """Get precision statistics."""
        return {
            "current_mode": self._current_mode.value,
            "avg_error": sum(self._error_history) / len(self._error_history) if self._error_history else 0.0,
            "max_error": max(self._error_history) if self._error_history else 0.0,
            "calibration_samples": len(self._error_history),
            "fallbacks": sum(1 for e in self._error_history if e > self.config.tolerance),
        }
