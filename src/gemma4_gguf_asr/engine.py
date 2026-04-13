"""
Gemma-4 GGUF + mmproj(libmtmd) 로컬 오디오 전사.

- PR #21421 기반 llama-cpp-python 빌드 필요(mtmd_support_audio=True).
- 스레드 안전: 동시 전사는 lock으로 직렬화.
"""

from __future__ import annotations

import ctypes
import threading
import time
from pathlib import Path
from typing import Any, Optional, Tuple


def pick_gemma4_gguf_pair(model_dir: Path) -> Tuple[Path, Path]:
    ggufs = list(Path(model_dir).glob("*.gguf"))
    main = next(
        (
            p
            for p in ggufs
            if p.name.lower().startswith("gemma-4") and "mmproj" not in p.name.lower()
        ),
        None,
    )
    mmproj = next((p for p in ggufs if "mmproj" in p.name.lower()), None)
    if not main or not mmproj:
        raise FileNotFoundError(
            f"GGUF/mmproj 없음: {model_dir} 후보={[p.name for p in ggufs]}"
        )
    return main, mmproj


def _transcribe_prompt_for_language(language: str, marker: str) -> str:
    lang = (language or "").strip().lower()
    if lang.startswith("ko") or "korean" in lang:
        return f"한국어 음성을 문자 그대로 전사하세요. 전사문만 출력하세요.\n{marker}\n"
    return f"Transcribe the speech. Output transcript only.\n{marker}\n"


class Gemma4GgufMtmdEngine:
    """Llama + mtmd 컨텍스트를 세션 동안 유지."""

    def __init__(
        self,
        *,
        model_dir: Path,
        n_ctx: int,
        n_gpu_layers: int,
        n_batch: int = 512,
    ) -> None:
        self._model_dir = Path(model_dir)
        self._n_ctx = int(n_ctx)
        self._n_gpu_layers = int(n_gpu_layers)
        self._n_batch = int(n_batch)
        self._lock = threading.Lock()
        self._llm = None
        self._mt = None
        self._main_gguf: Optional[Path] = None
        self._mmproj: Optional[Path] = None
        self._last_error: str = ""

    def _ensure_loaded(self) -> bool:
        if self._llm is not None and self._mt is not None:
            return True
        try:
            import llama_cpp.mtmd_cpp as m
            from llama_cpp import Llama

            self._main_gguf, self._mmproj = pick_gemma4_gguf_pair(self._model_dir)
            self._llm = Llama(
                str(self._main_gguf),
                n_gpu_layers=self._n_gpu_layers,
                n_ctx=self._n_ctx,
                n_batch=self._n_batch,
                verbose=False,
            )
            ctxp = m.mtmd_context_params_default()
            ctxp.use_gpu = bool(self._n_gpu_layers != 0)
            ctxp.print_timings = False
            self._mt = m.mtmd_init_from_file(
                str(self._mmproj).encode("utf-8"), self._llm.model, ctxp
            )
            if self._mt is None:
                self._last_error = "mtmd_init_from_file failed"
                self._llm = None
                return False
            if not m.mtmd_support_audio(self._mt):
                self._last_error = "mtmd_support_audio is False (llama.cpp PR #21421 빌드 필요)"
                m.mtmd_free(self._mt)
                self._mt = None
                self._llm = None
                return False
            self._last_error = ""
            return True
        except Exception as e:
            self._last_error = str(e)
            self._llm = None
            self._mt = None
            return False

    def health(self) -> dict:
        d = self._model_dir
        ok_dir = d.is_dir()
        out: dict = {
            "ok": False,
            "mode": "GEMMA4_GGUF_MTMD",
            "model_dir": str(d),
            "error": "",
        }
        if not ok_dir:
            out["error"] = "model_dir missing"
            return out
        try:
            pick_gemma4_gguf_pair(d)
        except FileNotFoundError as e:
            out["error"] = str(e)
            return out
        if not self._ensure_loaded():
            out["error"] = self._last_error or "load failed"
            return out
        import llama_cpp.mtmd_cpp as m

        out["ok"] = True
        out["main"] = self._main_gguf.name if self._main_gguf else ""
        out["mmproj"] = self._mmproj.name if self._mmproj else ""
        out["vision"] = m.mtmd_support_vision(self._mt)
        out["audio"] = m.mtmd_support_audio(self._mt)
        out["sample_rate"] = m.mtmd_get_audio_sample_rate(self._mt)
        return out

    def transcribe(
        self,
        wav_path: str,
        *,
        language: str = "Korean",
        max_seconds: float = 8.0,
        max_new_tokens: int = 256,
        bench: Optional[dict[str, Any]] = None,
    ) -> str:
        import numpy as np
        import soundfile as sf
        import llama_cpp.mtmd_cpp as m
        import llama_cpp.llama_cpp as L

        p = Path(wav_path)
        if not p.is_file():
            return ""

        with self._lock:
            if not self._ensure_loaded():
                return ""

            llm = self._llm
            mt = self._mt
            assert llm is not None and mt is not None

            data, sr = sf.read(str(p), dtype="float32", always_2d=False)
            if data.ndim > 1:
                data = data.mean(axis=1)
            n = min(len(data), int(max_seconds * sr))
            data = data[:n].astype(np.float32)
            target_sr = m.mtmd_get_audio_sample_rate(mt)
            if target_sr > 0 and sr != target_sr:
                try:
                    import torch
                    import torchaudio

                    t = torch.from_numpy(data).unsqueeze(0)
                    data = (
                        torchaudio.functional.resample(t, sr, target_sr)
                        .squeeze(0)
                        .numpy()
                        .astype(np.float32)
                    )
                    sr = target_sr
                except Exception:
                    pass

            input_audio_sec = float(len(data)) / float(sr) if sr and len(data) else 0.0

            arr = (ctypes.c_float * len(data))(*data.tolist())
            abm = m.mtmd_bitmap_init_from_audio(len(data), arr)
            if abm is None:
                return ""

            marker = m.mtmd_default_marker().decode("utf-8")
            prompt = _transcribe_prompt_for_language(language, marker)
            it = m.mtmd_input_text()
            it.text = prompt.encode("utf-8")
            it.add_special = True
            it.parse_special = True
            chunks = m.mtmd_input_chunks_init()
            ba = (m.mtmd_bitmap_p_ctypes * 1)(abm)
            rc = m.mtmd_tokenize(mt, chunks, ctypes.byref(it), ba, 1)
            if rc != 0:
                m.mtmd_input_chunks_free(chunks)
                m.mtmd_bitmap_free(abm)
                return ""

            t_prefill0 = time.perf_counter()
            llm.reset()
            llm._ctx.kv_cache_clear()
            n_chunks = m.mtmd_input_chunks_size(chunks)
            for i in range(n_chunks):
                chunk = m.mtmd_input_chunks_get(chunks, i)
                if chunk is None:
                    continue
                ctype = m.mtmd_input_chunk_get_type(chunk)
                if ctype == m.MTMD_INPUT_CHUNK_TYPE_TEXT:
                    n_tok = ctypes.c_size_t()
                    toks = m.mtmd_input_chunk_get_tokens_text(chunk, ctypes.byref(n_tok))
                    if toks and n_tok.value > 0:
                        llm.eval([toks[j] for j in range(n_tok.value)])
                elif ctype in (m.MTMD_INPUT_CHUNK_TYPE_IMAGE, m.MTMD_INPUT_CHUNK_TYPE_AUDIO):
                    has_following_text = False
                    for j in range(i + 1, n_chunks):
                        c2 = m.mtmd_input_chunks_get(chunks, j)
                        if c2 is None:
                            continue
                        if m.mtmd_input_chunk_get_type(c2) != m.MTMD_INPUT_CHUNK_TYPE_TEXT:
                            has_following_text = True
                            break
                        n2 = ctypes.c_size_t()
                        tp = m.mtmd_input_chunk_get_tokens_text(c2, ctypes.byref(n2))
                        if tp and n2.value > 0:
                            has_following_text = True
                            break
                    logits_last = not has_following_text
                    n_before = llm.n_tokens
                    new_past = L.llama_pos(0)
                    er = m.mtmd_helper_eval_chunk_single(
                        mt,
                        llm._ctx.ctx,
                        chunk,
                        L.llama_pos(llm.n_tokens),
                        L.llama_seq_id(0),
                        llm.n_batch,
                        logits_last,
                        ctypes.byref(new_past),
                    )
                    if er != 0:
                        m.mtmd_input_chunks_free(chunks)
                        m.mtmd_bitmap_free(abm)
                        return ""
                    llm.n_tokens = new_past.value
                    ph = 256000 if ctype == m.MTMD_INPUT_CHUNK_TYPE_AUDIO else 255999
                    for pos in range(n_before, llm.n_tokens):
                        llm.input_ids[pos] = ph

            m.mtmd_input_chunks_free(chunks)
            m.mtmd_bitmap_free(abm)

            turn = "<start_of_turn>model\n"
            llm.eval(llm.tokenize(turn.encode("utf-8"), add_bos=False, special=True))
            if bench is not None:
                # 디코딩 직전 KV에 쌓인 시퀀스 길이(프롬프트+오디오+turn 접두)
                bench["n_lm_positions_after_prefill"] = int(llm.n_tokens)
            t_gen0 = time.perf_counter()

            pieces: list[int] = []
            for token in llm.generate([], reset=False, temp=0.0, top_k=1):
                if L.llama_vocab_is_eog(llm._model.vocab, token):
                    break
                pieces.append(token)
                if len(pieces) >= max_new_tokens:
                    break
                if llm.detokenize(pieces).endswith(b"<end_of_turn>"):
                    break
            t_gen1 = time.perf_counter()
            if bench is not None:
                prefill_sec = t_gen0 - t_prefill0
                gen_sec = max(t_gen1 - t_gen0, 1e-9)
                n_out = len(pieces)
                bench["input_audio_sec"] = input_audio_sec
                bench["input_sample_rate"] = int(sr)
                bench["input_samples"] = int(len(data))
                bench["prefill_sec"] = prefill_sec
                bench["generate_sec"] = gen_sec
                bench["n_output_tokens"] = n_out
                bench["tokens_per_sec"] = n_out / gen_sec
                asec = max(input_audio_sec, 1e-9)
                bench["prefill_rtf"] = prefill_sec / asec
                bench["prefill_audio_x_realtime"] = asec / max(prefill_sec, 1e-9)
                bench["n_lm_positions_after_decode"] = int(
                    bench.get("n_lm_positions_after_prefill", 0)
                ) + int(n_out)

            raw = llm.detokenize(pieces)
            if raw.endswith(b"<end_of_turn>"):
                raw = raw[: -len(b"<end_of_turn>")]
            return raw.decode("utf-8", errors="replace").strip()
