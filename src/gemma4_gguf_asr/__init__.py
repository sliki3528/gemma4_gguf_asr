"""
Gemma 4 GGUF + mmproj (libmtmd) 기반 WAV 전사.
"""

from .engine import Gemma4GgufMtmdEngine, pick_gemma4_gguf_pair

__all__ = ["Gemma4GgufMtmdEngine", "pick_gemma4_gguf_pair"]
__version__ = "0.1.1"
