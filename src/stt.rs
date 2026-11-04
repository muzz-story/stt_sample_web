//! Deepgram 스트리밍 결과(JSON) → 화자별 발언 목록.

use serde_json::{Value, json};

#[derive(Debug, PartialEq)]
pub struct Utterance {
    pub speaker: u64,
    pub start: f64,
    pub text: String,
}

#[derive(Default)]
pub struct Transcript {
    pub finals: Vec<Utterance>,
    pub interim: Vec<Utterance>,
}

impl Transcript {
    /// Deepgram 메시지 하나를 반영. 화면 갱신이 필요하면 true.
    pub fn apply(&mut self, msg: &str) -> bool {
        let Ok(v) = serde_json::from_str::<Value>(msg) else { return false };
        if v["type"] != "Results" {
            return false;
        }
        // 단어 단위 speaker 를 연속 구간으로 묶는다.
        let mut segs: Vec<Utterance> = Vec::new();
        for w in v["channel"]["alternatives"][0]["words"].as_array().into_iter().flatten() {
            let word = w["punctuated_word"].as_str().or(w["word"].as_str()).unwrap_or_default();
            let speaker = w["speaker"].as_u64().unwrap_or(0);
            match segs.last_mut() {
                Some(s) if s.speaker == speaker => {
                    s.text.push(' ');
                    s.text.push_str(word);
                }
                _ => segs.push(Utterance { speaker, start: w["start"].as_f64().unwrap_or(0.0), text: word.into() }),
            }
        }
        if v["is_final"].as_bool() != Some(true) {
            self.interim = segs;
            return true;
        }
        self.interim.clear();
        // 같은 화자가 이어서 말하면 한 발언으로 합친다.
        for s in segs {
            match self.finals.last_mut() {
                Some(last) if last.speaker == s.speaker => {
                    last.text.push(' ');
                    last.text.push_str(&s.text);
                }
                _ => self.finals.push(s),
            }
        }
        true
    }

    /// VibeVoice `/v1/stream` 응답: 매 청크마다 누적 segments 전체가 온다.
    /// `{"segments":[{"Start":0.0,"End":2.1,"Speaker":"0","Content":"..."}]}` / `{"error":"..."}`
    pub fn apply_segments(&mut self, msg: &str) -> Result<bool, String> {
        let Ok(v) = serde_json::from_str::<Value>(msg) else { return Ok(false) };
        if let Some(e) = v["error"].as_str() {
            return Err(e.into());
        }
        let Some(segs) = v["segments"].as_array() else { return Ok(false) };
        self.finals = segs
            .iter()
            .map(|s| Utterance {
                // "?" (화자 미상) 은 0 으로
                speaker: s["Speaker"].as_str().and_then(|x| x.parse().ok()).or(s["Speaker"].as_u64()).unwrap_or(0),
                start: s["Start"].as_f64().unwrap_or(0.0),
                text: s["Content"].as_str().unwrap_or_default().into(),
            })
            .collect();
        self.interim.clear();
        Ok(true)
    }

    pub fn to_json(&self) -> String {
        let list = |v: &[Utterance]| -> Vec<Value> {
            v.iter().map(|u| json!({ "speaker": u.speaker, "start": u.start, "text": u.text })).collect()
        };
        json!({ "type": "transcript", "finals": list(&self.finals), "interim": list(&self.interim) }).to_string()
    }
}

/// shortcut: 선형 보간 리샘플러(저역통과 필터 없음). 16k→24k 업샘플 용도라 충분, 다운샘플에 쓰게 되면 rubato 등으로 교체.
pub struct Resampler {
    step: f64, // 출력 1샘플당 입력 샘플 수
    pos: f64,
    prev: f32,
}

impl Resampler {
    pub fn new(in_rate: u32, out_rate: u32) -> Self {
        Self { step: in_rate as f64 / out_rate as f64, pos: 0.0, prev: 0.0 }
    }

    pub fn push(&mut self, input: impl Iterator<Item = f32>, out: &mut Vec<f32>) {
        for x in input {
            while self.pos < 1.0 {
                out.push(self.prev + (x - self.prev) * self.pos as f32);
                self.pos += self.step;
            }
            self.pos -= 1.0;
            self.prev = x;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resamples_across_chunks() {
        for out_rate in [16_000, 24_000] {
            let (mut rs, mut out) = (Resampler::new(16_000, out_rate), Vec::new());
            for _ in 0..10 {
                rs.push((0..1600).map(|_| 0.5), &mut out); // 100ms 씩 10번 = 1초
            }
            assert!((out.len() as i64 - out_rate as i64).abs() <= 1, "{out_rate}: {}", out.len());
        }
    }

    #[test]
    fn parses_vibevoice_segments() {
        let mut t = Transcript::default();
        let msg = r#"{"chunk":3,"text":"..","segments":[
            {"Start":0.0,"End":2.0,"Speaker":"0","Content":"안녕하세요."},
            {"Start":2.0,"End":4.0,"Speaker":"1","Content":"네, 반갑습니다."},
            {"Start":4.0,"End":5.0,"Speaker":"?","Content":"음"}]}"#;
        assert_eq!(t.apply_segments(msg), Ok(true));
        assert_eq!(t.finals.iter().map(|u| u.speaker).collect::<Vec<_>>(), vec![0, 1, 0]);
        assert_eq!(t.finals[1].text, "네, 반갑습니다.");
        assert_eq!(t.apply_segments(r#"{"error":"boom"}"#), Err("boom".into()));
    }

    fn result(is_final: bool, words: &[(u64, f64, &str)]) -> String {
        let words: Vec<Value> =
            words.iter().map(|(s, t, w)| json!({ "word": w, "punctuated_word": w, "start": t, "speaker": s })).collect();
        json!({ "type": "Results", "is_final": is_final, "channel": { "alternatives": [{ "words": words }] } }).to_string()
    }

    #[test]
    fn groups_and_merges_by_speaker() {
        let mut t = Transcript::default();
        assert!(!t.apply(r#"{"type":"Metadata"}"#));

        t.apply(&result(false, &[(0, 1.0, "안녕")]));
        assert_eq!(t.interim.len(), 1);
        assert!(t.finals.is_empty());

        t.apply(&result(true, &[(0, 1.0, "안녕"), (0, 1.4, "하세요."), (1, 2.5, "네,")]));
        assert!(t.interim.is_empty());
        assert_eq!(t.finals, vec![
            Utterance { speaker: 0, start: 1.0, text: "안녕 하세요.".into() },
            Utterance { speaker: 1, start: 2.5, text: "네,".into() },
        ]);

        t.apply(&result(true, &[(1, 3.0, "반갑습니다.")]));
        assert_eq!(t.finals.len(), 2);
        assert_eq!(t.finals[1].text, "네, 반갑습니다.");
    }
}
