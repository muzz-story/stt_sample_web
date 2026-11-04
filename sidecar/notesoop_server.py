"""Notesoop-STT-Ko-v1 로컬 변환 서버 (녹음 종료 후 일괄 변환).

Rust 서버가 녹음이 끝나면 전체 오디오(16kHz mono PCM16)를 POST 하고,
여기서 무음 기준으로 발언을 나눠 인식한 뒤 시간별 segments 로 돌려준다.
Notesoop 은 화자 분리를 지원하지 않아 모두 "화자 1" 로 나온다.

설치 (NVIDIA GPU 권장, VRAM 약 1.5GB, ffmpeg 필요):
  git clone https://huggingface.co/Phsoft-ai/Notesoop-STT-Ko-v1
  pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu128
  pip install -r Notesoop-STT-Ko-v1/requirements.txt fastapi uvicorn pillow   # pillow·torchvision: embeddinggemma-2 프로세서가 요구
  pip install https://github.com/kpu/kenlm/archive/master.zip   # 선택 (Windows 는 빌드 도구 필요, 없으면 --no-kenlm)
실행:
  python sidecar/notesoop_server.py --model-dir Notesoop-STT-Ko-v1 [--no-kenlm] [--port 7871]

API:
  GET  /v1/config          → {"model", "sample_rate"}
  POST /v1/transcribe_pcm  body = little-endian int16 PCM (16kHz mono)
                           → {"segments": [{"Start","End","Speaker","Content"}...]}
"""
import os
import sys
import tempfile
import wave

import numpy as np

RATE = 16000
FRAME = RATE // 10       # 0.1초 단위로 판정
SILENCE_RMS = 0.01       # 이보다 조용하면 무음
SILENCE_FRAMES = 6       # 0.6초 무음이면 발언 끝
MAX_FRAMES = 150         # 한 발언 최대 15초 (모델 chunk 16초 이내)
PAD = RATE // 5          # 발언 앞뒤 0.2초 여유


def split_utterances(audio: np.ndarray) -> list[tuple[int, int]]:
    """무음 기준으로 발언 구간 (start, end) 샘플 인덱스 목록."""
    n = len(audio) // FRAME
    voiced = [np.sqrt(np.mean(audio[i * FRAME:(i + 1) * FRAME] ** 2)) >= SILENCE_RMS for i in range(n)]
    spans, start, quiet = [], None, 0
    for i, v in enumerate(voiced):
        if start is None:
            if v:
                start, quiet = i, 0
            continue
        quiet = 0 if v else quiet + 1
        if quiet >= SILENCE_FRAMES or i + 1 - start >= MAX_FRAMES:
            spans.append((start, i + 1 - quiet))
            start = None
    if start is not None:
        spans.append((start, n - quiet))
    return [(max(0, s * FRAME - PAD), min(len(audio), e * FRAME + PAD)) for s, e in spans]


def main():
    import argparse
    import asyncio

    import torch
    import uvicorn
    from fastapi import FastAPI, Request

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="git clone 한 Notesoop-STT-Ko-v1 폴더")
    parser.add_argument("--no-kenlm", action="store_true")
    parser.add_argument("--port", type=int, default=7871)
    args = parser.parse_args()

    sys.path.insert(0, os.path.abspath(args.model_dir))
    os.chdir(args.model_dir)  # 체크포인트를 상대 경로로 읽는 경우 대비
    from run_inference import EmbeddingGemmaSTT

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Notesoop 로딩 중 (device={device}, kenlm={not args.no_kenlm})")
    stt = EmbeddingGemmaSTT(device=device, use_kenlm=not args.no_kenlm)
    gpu = asyncio.Lock()  # 요청이 겹쳐도 GPU 추론은 하나씩

    def transcribe(audio: np.ndarray) -> str:
        # 공개된 API 가 파일 경로만 받으므로 임시 WAV 로 넘긴다
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            with wave.open(path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(RATE)
                w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
            text, _rtf = stt.transcribe(path, chunk_sec=16.0, overlap_sec=1.0, batch_size=1)
            return text.strip()
        finally:
            os.remove(path)

    app = FastAPI()

    @app.get("/v1/config")
    def config():
        return {"model": "Notesoop-STT-Ko-v1", "sample_rate": RATE}

    @app.post("/v1/transcribe_pcm")
    async def transcribe_pcm(request: Request):
        audio = np.frombuffer(await request.body(), "<i2").astype(np.float32) / 32768
        segments = []
        async with gpu:
            for s, e in split_utterances(audio):
                text = await asyncio.to_thread(transcribe, audio[s:e])
                if text:
                    segments.append({"Start": round(s / RATE, 2), "End": round(e / RATE, 2),
                                     "Speaker": "0", "Content": text})
        print(f"변환 완료: {len(audio) / RATE:.1f}초 → {len(segments)}개 발언")
        return {"segments": segments}

    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
