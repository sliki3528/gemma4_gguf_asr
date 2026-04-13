# gemma4_gguf_asr

- gemma4 gguf를 사용해 wav 파일을 전사하는 파이썬 모듈입니다. 

- A Python module for transcribing audio files to text using Gemma 4 GGUF and mmproj.


## Performance Test (Benchmark)

Testing environment: **Gemma 4 E2B** model with a **120s (16kHz, Mono) WAV file**.

| Metric | NVIDIA RTX 3090 Ti | NVIDIA RTX 3060 |
| :--- | :---: | :---: |
| **Prefill (Audio processing)** | 20.42s | 21.08s |
| **Generation (Text output)** | 16.24 tok/s | 16.62 tok/s |
| **Total elapsed time** | 84.7s | 84.0s |

### Technical Note
* The performance (tokens per second) is nearly identical between the RTX 3090 Ti and RTX 3060.
* This suggests that the current bottleneck may reside in the audio-processing optimization of the engine (llama.cpp) rather than hardware limitations.
