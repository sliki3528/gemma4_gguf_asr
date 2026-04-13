# gemma4-gguf-asr

특정 애플리케이션과 묶이지 않은 **배포용** Python 패키지입니다. 필요하면 가져다 쓰면 됩니다.

Gemma 4 **GGUF + mmproj** 폴더에서 **WAV → 텍스트** 전사를 수행합니다.  
`llama-cpp-python`의 **libmtmd** 오디오 경로가 켜진 빌드가 필요합니다 (llama.cpp PR #21421 계열).

## 프로젝트 성격 (유지보수)

**이슈·기능 요청에 답변하거나 우선순위를 잡을 계획은 없습니다.** 필요한 사람이 있으면 쓰고, **포크해서 마음대로 수정·배포·이름 바꿔 재공개**해도 됩니다. 고도화는 원작자보다 다른 사람이 더 잘할 수 있다고 보고, **그냥 올려만 둔 레퍼런스 구현**에 가깝습니다.

법적 허용 범위는 **[LICENSE](LICENSE) (MIT)** 를 따릅니다. 상업적 이용·수정·재배포·명칭 변경 모두 라이선스 조건만 지키면 됩니다.

## 설치

### 가상환경(권장)

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### 이 저장소 클론 후 (로컬 폴더에서 editable)

`gemma4_gguf_asr` 루트( `pyproject.toml` 이 있는 디렉터리)로 이동한 뒤:

```bash
cd path/to/gemma4_gguf_asr
python -m pip install -U pip
python -m pip install -e .
```

선택: WAV 샘플레이트가 모델과 다를 때 **torchaudio 리샘플**을 쓰려면:

```bash
python -m pip install -e ".[resample]"
```

### Git URL로 직접 설치

원격 저장소 주소만 바꿔서 사용합니다.

```bash
python -m pip install git+https://github.com/YOUR_ORG/gemma4-gguf-asr.git
# extras 포함
python -m pip install "gemma4-gguf-asr[resample] @ git+https://github.com/YOUR_ORG/gemma4-gguf-asr.git"
```

다른 저장소 안에 이 패키지가 **하위 폴더**로만 있을 때:

```bash
python -m pip install "git+https://github.com/YOUR_ORG/parent-repo.git#subdirectory=gemma4_gguf_asr"
```

### `llama-cpp-python` (Gemma4 오디오/mtmd 빌드)

PyPI 기본 휠은 환경에 따라 **Gemma4 오디오(mtmd)** 가 비활성일 수 있습니다.  
이 경우 `llama.cpp`(PR #21421 계열)로 **직접 빌드한** `llama-cpp-python`이 필요합니다.

설치가 제대로 되었는지 아래로 확인할 수 있습니다.

```bash
python -c "from llama_cpp import llama_cpp as L; print(bool(L.llama_supports_gpu_offload()))"
python -c "from llama_cpp import mtmd_cpp as m; print('mtmd imported')"
```

Windows에서 CUDA 빌드 시, 최신 Visual Studio 조합에서는 nvcc가 호스트 컴파일러 버전 체크로 실패할 수 있어
`--allow-unsupported-compiler` 플래그가 필요할 때가 있습니다.

```bash
# 예시(환경에 맞게 수정):
# - CUDA 켜기: -DGGML_CUDA=ON
# - 아키텍처 고정: -DCMAKE_CUDA_ARCHITECTURES=86-real (RTX 30xx)
# - VS 버전 우회: -DCMAKE_CUDA_FLAGS=--allow-unsupported-compiler
python -m pip install llama-cpp-python --force-reinstall --no-cache-dir
```

GPU·CUDA 아키텍처 고정 등은 `llama-cpp-python`·`llama.cpp` 쪽 빌드 가이드를 참고하면 됩니다.

## 사용

```python
from pathlib import Path
from gemma4_gguf_asr import Gemma4GgufMtmdEngine

eng = Gemma4GgufMtmdEngine(
    model_dir=Path("/path/to/gemma-4-E4B-it-GGUF"),
    n_ctx=8192,
    n_gpu_layers=-1,
    n_batch=512,
)
print(eng.health())
text = eng.transcribe("clip.wav", language="Korean", max_seconds=8.0)
print(text)
```

## 모델 파일

`model_dir` 안에 다음이 있어야 합니다.

- 메인 가중치: 이름이 `gemma-4`로 시작하고 `mmproj`가 **아닌** `.gguf`
- 멀티모달 프로젝션: 이름에 `mmproj`가 들어간 `.gguf`

## 라이선스

MIT — 전문은 `[LICENSE](LICENSE)` 파일을 참고하세요. 저작권 표시만 유지하면 자유롭게 사용·수정·재배포할 수 있습니다.