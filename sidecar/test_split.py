"""split_utterances 자체 점검: python sidecar/test_split.py"""
import numpy as np

from notesoop_server import PAD, RATE, split_utterances

tone = lambda sec: 0.3 * np.sin(np.arange(int(sec * RATE)) * 2 * np.pi * 220 / RATE).astype(np.float32)
silence = lambda sec: np.zeros(int(sec * RATE), np.float32)

# 무음 1초 / 말 2초 / 무음 1초 / 말 1초 → 발언 2개
audio = np.concatenate([silence(1), tone(2), silence(1), tone(1)])
spans = split_utterances(audio)
assert len(spans) == 2, spans
assert abs(spans[0][0] / RATE - 0.8) < 0.11 and abs(spans[0][1] / RATE - 3.2) < 0.11, spans  # 앞뒤 0.2초 여유
assert abs(spans[1][0] / RATE - 3.8) < 0.11, spans

# 쉬지 않고 40초 → 15초 단위로 잘림
spans = split_utterances(tone(40))
assert len(spans) == 3 and all((e - s) <= 15 * RATE + 2 * PAD for s, e in spans), spans

# 전부 무음 → 없음
assert split_utterances(silence(5)) == []
print("ok")
