# gemma4_gguf_asr

- gemma4 gguf를 사용해 wav 파일을 전사하는 파이썬 모듈입니다. 

- A Python module for transcribing audio files to text using Gemma 4 GGUF and mmproj.


## Performance Test (Benchmark)

Testing environment: **Gemma 4 E2B** model with a **120s (16kHz, Mono) WAV file**.

| Metric | NVIDIA RTX 3090 Ti | NVIDIA RTX 3060 |
| :--- | :---: | :---: |
| **Prefill (Audio processing)** | 1.5651s | 3.3417s |
| **Generation (Text output)** | 86.41 tok/s | 33.52 tok/s |
| **Total elapsed time** | 14.656s | 35.078s |
