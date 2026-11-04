# AI 면접 실시간 기록 샘플 (Rust)

브라우저에서 마이크(+시스템 오디오)를 녹음해 **MP3로 저장**하고, 선택한 음성 모델로 **화자별 기록**을 화면에 보여주는 샘플 앱입니다.

```
브라우저 (마이크/시스템 오디오, 16kHz PCM)
   │ WebSocket
   ▼
Rust 서버 (axum) ── MP3 저장 (recordings/)
   │
   ├─ Deepgram            클라우드, 실시간, 화자 구분
   ├─ VibeVoice-ASR       GPU 서버(vLLM), 실시간, 화자 구분
   └─ Notesoop-STT-Ko     로컬 Python 서버, 녹음 종료 후 변환
```

## 실행

준비: Rust 1.92+, Chrome

```powershell
cargo run
```

http://localhost:3030 을 Chrome에서 엽니다. 포트를 바꾸려면 `$env:PORT = "8080"`.

## 사용법

1. ⚙ 버튼 → 음성 모델 선택 → 저장
2. **● 녹음 시작** → 마이크 권한 허용
3. 녹음 중
   - **일시정지 / 재개**: 일시정지한 구간은 녹음과 타이머에서 빠집니다
   - **음소거**: 시간은 흐르고 무음으로 기록됩니다
   - **시스템 오디오**: 공유할 탭/화면을 고르고 **"오디오 공유"를 체크**하면 스피커 소리(예: AI 면접관 음성)도 함께 녹음됩니다
4. **■ 녹음 종료** → `recordings/interview_<시각>.mp3` 저장 (하단 상태 바에 길이·크기 표시)

**기존 음성 파일로 테스트**: 녹음 버튼 옆 📄 버튼 → mp3 / wav / m4a / flac / ogg 선택. 브라우저가 16kHz로 변환해 녹음과 같은 경로로 선택한 모델에 보냅니다 (변환본도 `recordings/`에 MP3로 저장). 기록 위에 원본 재생기가 나오고, **발언을 클릭하면 그 시점부터 재생**됩니다.

## 음성 모델

| 모델 | 방식 | 설정에 입력 | 비고 |
|---|---|---|---|
| Deepgram | 실시간 · 화자 구분 | API 키 | 비우면 서버의 `DEEPGRAM_API_KEY` 환경변수 사용 |
| VibeVoice-ASR-Streaming | 실시간 · 화자 구분 | 서버 주소 (예: `http://gpu-server:8000`) | VRAM 24GB 이상 Linux GPU 서버에서 [vLLM 서버](https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-vllm-asr-streaming.md) 실행 필요 |
| Notesoop-STT-Ko | 녹음 종료 후 변환 · 화자 구분 없음 | 서버 주소 (기본 `http://localhost:7871`) | 아래 설치 필요, VRAM 약 3GB |
| 사용 안 함 | 녹음만 | — | |

연결에 실패해도 녹음은 계속되고, 이유가 상태 바에 표시됩니다.

### Notesoop 로컬 서버 설치 (Windows, NVIDIA GPU)

```powershell
git clone https://huggingface.co/Phsoft-ai/Notesoop-STT-Ko-v1 models/Notesoop-STT-Ko-v1
uv venv .venv-notesoop --python 3.12
uv pip install --python .venv-notesoop/Scripts/python.exe torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-notesoop/Scripts/python.exe -r models/Notesoop-STT-Ko-v1/requirements.txt fastapi uvicorn pillow
```

실행 (첫 실행 시 기반 모델 `google/embeddinggemma-2` 약 745MB 자동 다운로드):

```powershell
.venv-notesoop\Scripts\python.exe sidecar\notesoop_server.py --model-dir models\Notesoop-STT-Ko-v1 --no-kenlm
```

KenLM 언어모델을 설치하면 `--no-kenlm` 없이 실행해 정확도를 높일 수 있습니다. 반드시 위 가상환경에 설치하고(시스템 `pip`가 아님), C++ 빌드 도구(Visual Studio Build Tools)가 필요합니다:

```powershell
uv pip install --python .venv-notesoop/Scripts/python.exe https://github.com/kpu/kenlm/archive/master.zip
```

## 코드 구성

| 파일 | 역할 |
|---|---|
| [src/main.rs](src/main.rs) | 웹 서버, WebSocket 세션, MP3 인코딩(LAME 32kbps), 모델별 연결 |
| [src/stt.rs](src/stt.rs) | 모델 응답 → 화자별 발언 목록, 16k→24k 리샘플러 |
| [static/index.html](static/index.html) | 화면 전체 (오디오 캡처, 파형, 설정, 기록 목록) — 빌드 시 바이너리에 포함 |
| [sidecar/notesoop_server.py](sidecar/notesoop_server.py) | Notesoop 변환 서버 (무음 기준으로 발언을 나눠 인식) |

`static/index.html`을 수정하면 `cargo run`으로 다시 빌드해야 반영됩니다.

## 테스트

```powershell
cargo test                       # 기록 파싱, 리샘플러
python sidecar\test_split.py     # 발언 분할 (numpy 필요)
```

## 제한 사항

- Chrome 기준입니다 (16kHz AudioContext 리샘플링). Firefox는 동작하지 않을 수 있습니다.
- 마이크와 시스템 오디오를 한 채널로 섞어 보내므로, 화자 구분은 모델의 판단에 따릅니다.
- Notesoop 모드는 녹음 전체를 메모리에 보관합니다 (1시간 ≈ 115MB).
- 요약 보기, 기록 텍스트 파일 저장은 없습니다.
