//! 브라우저(마이크/시스템 오디오) → WebSocket(16kHz PCM16) → 이 서버 → MP3 저장 + STT 엔진 → 브라우저에 실시간 기록 전송.
//!
//! STT 엔진 (브라우저가 첫 메시지로 선택):
//! - deepgram : Deepgram 클라우드 (API 키)
//! - vibevoice: VibeVoice-ASR-Streaming vLLM 서버 (`/v1/config`, `/v1/stream` 프로토콜, 실시간)
//! - notesoop : sidecar/notesoop_server.py (`/v1/transcribe_pcm`, 녹음 종료 후 일괄 변환)

mod stt;

use axum::Router;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::response::Html;
use axum::routing::get;
use futures_util::stream::{SplitSink, SplitStream};
use futures_util::{SinkExt, StreamExt};
use mp3lame_encoder::{Bitrate, Builder, FlushNoGap, MonoPcm, Quality};
use serde_json::{Value, json};
use std::io::Write;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use stt::{Resampler, Transcript};
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::{self, Message as SttMsg, client::IntoClientRequest};
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

type Error = Box<dyn std::error::Error + Send + Sync>;
type SttWs = WebSocketStream<MaybeTlsStream<TcpStream>>;

const RATE: u32 = 16_000;
const DG_URL: &str = "wss://api.deepgram.com/v1/listen?model=nova-2&language=ko&encoding=linear16\
    &sample_rate=16000&channels=1&diarize=true&interim_results=true&punctuate=true&smart_format=true";

/// 16kHz mono PCM → MP3(32kbps CBR) 파일. WAV(256kbps) 대비 약 1/8 크기.
struct Mp3File {
    enc: mp3lame_encoder::Encoder,
    file: std::fs::File,
    buf: Vec<u8>,
}

impl Mp3File {
    fn create(path: &str) -> Result<Self, Error> {
        let mut b = Builder::new().ok_or("LAME 초기화 실패")?;
        b.set_num_channels(1)?;
        b.set_sample_rate(RATE)?;
        b.set_brate(Bitrate::Kbps32)?;
        b.set_quality(Quality::Good)?;
        Ok(Self { enc: b.build()?, file: std::fs::File::create(path)?, buf: Vec::new() })
    }

    fn write(&mut self, pcm: &[i16]) -> Result<(), Error> {
        self.buf.clear();
        self.buf.reserve(pcm.len() * 5 / 4 + 7200); // LAME 권장 최대 출력 크기
        self.enc.encode_to_vec(MonoPcm(pcm), &mut self.buf)?;
        Ok(self.file.write_all(&self.buf)?)
    }

    fn finish(mut self) -> Result<(), Error> {
        self.buf.clear();
        self.buf.reserve(7200);
        self.enc.flush_to_vec::<FlushNoGap>(&mut self.buf)?;
        Ok(self.file.write_all(&self.buf)?)
    }
}

/// 연결된 STT 엔진. Deepgram 과 VibeVoice 계열은 오디오 형식/메시지 형식만 다르다.
struct Stt {
    tx: SplitSink<SttWs, SttMsg>,
    rx: SplitStream<SttWs>,
    /// None = Deepgram(16k PCM16), Some = VibeVoice 프로토콜(서버 샘플레이트 float32)
    resample: Option<Resampler>,
}

impl Stt {
    async fn send_audio(&mut self, pcm16: &[i16], raw: axum::body::Bytes) -> Result<(), tungstenite::Error> {
        let Some(rs) = self.resample.as_mut() else { return self.tx.send(SttMsg::Binary(raw)).await };
        let mut out = Vec::with_capacity(pcm16.len() * 2);
        rs.push(pcm16.iter().map(|&s| s as f32 / 32768.0), &mut out);
        let bytes: Vec<u8> = out.iter().flat_map(|s| s.to_le_bytes()).collect();
        self.tx.send(SttMsg::Binary(bytes.into())).await
    }

    /// 엔진 응답 하나를 기록에 반영. Err 는 엔진이 보낸 오류 메시지.
    fn apply(&self, tr: &mut Transcript, msg: &str) -> Result<bool, String> {
        if self.resample.is_none() { Ok(tr.apply(msg)) } else { tr.apply_segments(msg) }
    }
}

#[tokio::main]
async fn main() {
    let app = Router::new()
        .route("/", get(|| async { Html(include_str!("../static/index.html")) }))
        .route("/ws", get(|ws: WebSocketUpgrade| async { ws.on_upgrade(session) }));
    let port = std::env::var("PORT").unwrap_or_else(|_| "3030".into());
    let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{port}")).await.expect("포트가 사용 중입니다 (PORT 환경변수로 변경)");
    println!("http://localhost:{port} 에서 열어주세요");
    axum::serve(listener, app).await.unwrap();
}

async fn session(mut sock: WebSocket) {
    if let Err(e) = run_session(&mut sock).await {
        let _ = sock.send(status(&format!("오류: {e}"))).await;
    }
}

fn status(text: &str) -> Message {
    Message::Text(json!({ "type": "status", "text": text }).to_string().into())
}

enum Conn {
    /// 실시간 스트리밍 (Deepgram, VibeVoice)
    Stream(Stt),
    /// 녹음 종료 후 일괄 변환 (Notesoop): 서버 주소
    Batch(String),
}

/// GET {base}/v1/config — 모델 서버가 살아있는지 확인하고 설정을 읽는다.
async fn server_config(base: &str) -> Result<Value, String> {
    reqwest::get(format!("{base}/v1/config"))
        .await
        .and_then(|r| r.error_for_status())
        .map_err(|e| format!("{base} 연결 실패: {e}"))?
        .text()
        .await
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .ok_or_else(|| "/v1/config 응답 형식 오류".into())
}

/// 브라우저 설정 `{"engine": "...", "key": "...", "url": "..."}` 에 맞는 엔진에 연결.
async fn connect(cfg: &Value) -> Result<(Conn, String), String> {
    let field = |k: &str| cfg[k].as_str().map(str::trim).filter(|s| !s.is_empty()).map(String::from);
    let engine = cfg["engine"].as_str().unwrap_or_default();
    let (ws, resample, label) = if engine == "deepgram" {
        let key = field("key").or_else(|| std::env::var("DEEPGRAM_API_KEY").ok()).ok_or("Deepgram API 키를 입력하세요")?;
        let mut req = DG_URL.into_client_request().map_err(|e| e.to_string())?;
        req.headers_mut().insert("Authorization", format!("Token {key}").parse().map_err(|_| "API 키 형식 오류")?);
        let (ws, _) = tokio_tungstenite::connect_async(req).await.map_err(|e| format!("Deepgram 연결 실패: {e}"))?;
        (ws, None, "Deepgram".to_string())
    } else if engine == "notesoop" {
        let base = field("url").ok_or("모델 서버 주소를 입력하세요")?;
        let base = base.trim_end_matches('/').to_string();
        let conf = server_config(&base).await?;
        let model = conf["model"].as_str().unwrap_or(engine).to_string();
        return Ok((Conn::Batch(base), model));
    } else if engine == "vibevoice" {
        let base = field("url").ok_or("모델 서버 주소를 입력하세요")?;
        let base = base.trim_end_matches('/');
        let conf = server_config(base).await?;
        let rate = conf["sample_rate"].as_u64().ok_or("/v1/config 에 sample_rate 없음")? as u32;
        let ws_url = format!("{}/v1/stream", base.replacen("http", "ws", 1)); // https → wss
        let (mut ws, _) = tokio_tungstenite::connect_async(ws_url).await.map_err(|e| format!("스트림 연결 실패: {e}"))?;
        ws.send(SttMsg::text("{}")).await.map_err(|e| e.to_string())?; // 첫 메시지 = 세션 옵션(JSON)
        let model = conf["model"].as_str().unwrap_or(engine).to_string();
        (ws, Some(Resampler::new(RATE, rate)), model)
    } else {
        return Err("STT 엔진 미선택".into());
    };
    let (tx, rx) = ws.split();
    Ok((Conn::Stream(Stt { tx, rx, resample }), label))
}

/// Notesoop: 녹음 전체를 보내 화자/시간별 segments 를 받는다 (VibeVoice 와 같은 segments 형식).
async fn transcribe_batch(base: &str, pcm: &[i16]) -> Result<String, String> {
    let body: Vec<u8> = pcm.iter().flat_map(|s| s.to_le_bytes()).collect();
    reqwest::Client::new()
        .post(format!("{base}/v1/transcribe_pcm"))
        .body(body)
        .send()
        .await
        .and_then(|r| r.error_for_status())
        .map_err(|e| format!("변환 요청 실패: {e}"))?
        .text()
        .await
        .map_err(|e| e.to_string())
}

async fn next_msg(stt: &mut Option<Stt>) -> Option<Result<SttMsg, tungstenite::Error>> {
    match stt {
        Some(s) => s.rx.next().await,
        None => std::future::pending().await,
    }
}

async fn run_session(sock: &mut WebSocket) -> Result<(), Error> {
    // 브라우저의 첫 메시지 = 엔진 설정
    let cfg: Value = match sock.recv().await {
        Some(Ok(Message::Text(t))) => serde_json::from_str(t.as_str())?,
        _ => return Ok(()),
    };
    std::fs::create_dir_all("recordings")?;
    let path = format!("recordings/interview_{}.mp3", SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs());
    let mut mp3 = Mp3File::create(&path)?;

    let (mut stt, mut batch): (Option<Stt>, Option<(String, Vec<i16>)>) = (None, None);
    let conn = connect(&cfg).await;
    println!("[{path}] 시작 engine={} → {}", cfg["engine"], conn.as_ref().map_or_else(|e| e.as_str(), |(_, l)| l.as_str()));
    match conn {
        Ok((Conn::Stream(s), label)) => {
            sock.send(status(&format!("{label} 연결됨 · 실시간 기록 중"))).await?;
            stt = Some(s);
        }
        Ok((Conn::Batch(url), label)) => {
            sock.send(status(&format!("{label} 연결됨 · 녹음을 종료하면 변환합니다"))).await?;
            // shortcut: 녹음 전체 PCM 을 메모리에 보관(1시간 ≈ 115MB). 더 긴 녹음이 필요하면 임시 파일로.
            batch = Some((url, Vec::new()));
        }
        Err(e) => sock.send(status(&format!("{e} — 녹음만 진행합니다"))).await?,
    }
    let mut total = 0usize; // 녹음된 샘플 수
    let mut tr = Transcript::default();
    let mut keepalive = tokio::time::interval(Duration::from_secs(5)); // 일시정지 중 Deepgram 10초 타임아웃 방지

    let stopped = loop {
        tokio::select! {
            msg = sock.recv() => match msg {
                Some(Ok(Message::Binary(raw))) => {
                    let pcm: Vec<i16> = raw.chunks_exact(2).map(|s| i16::from_le_bytes([s[0], s[1]])).collect();
                    mp3.write(&pcm)?;
                    total += pcm.len();
                    if let Some((_, all)) = batch.as_mut() {
                        all.extend_from_slice(&pcm);
                    }
                    if let Some(s) = stt.as_mut() {
                        let _ = s.send_audio(&pcm, raw).await; // 끊김은 아래 수신 쪽에서 감지
                    }
                }
                Some(Ok(Message::Text(t))) if t.as_str() == "stop" => break true,
                Some(Ok(_)) => {}
                _ => break false, // 브라우저가 그냥 닫힘
            },
            msg = next_msg(&mut stt) => match msg {
                Some(Ok(SttMsg::Text(t))) => match stt.as_ref().map(|s| s.apply(&mut tr, t.as_str())) {
                    Some(Ok(true)) => sock.send(Message::Text(tr.to_json().into())).await?,
                    Some(Err(e)) => sock.send(status(&format!("STT 오류: {e}"))).await?,
                    _ => {}
                },
                Some(Ok(_)) => {}
                _ => {
                    stt = None;
                    sock.send(status("STT 연결 끊김 — 녹음은 계속됩니다")).await?;
                }
            },
            _ = keepalive.tick() => if let Some(s) = stt.as_mut().filter(|s| s.resample.is_none()) {
                let _ = s.tx.send(SttMsg::text(r#"{"type":"KeepAlive"}"#)).await;
            },
        }
    };

    mp3.finish()?;
    let secs = total as f64 / RATE as f64;
    println!("[{path}] {:.1}초 수신 → {}", secs, if stopped { "종료 요청" } else { "브라우저 연결 끊김 (변환 안 함)" });
    if !stopped {
        return Ok(());
    }
    // 종료 요청: 엔진에 끝을 알리고 남은 결과를 받는다.
    if let Some(mut s) = stt {
        let end = if s.resample.is_none() { r#"{"type":"CloseStream"}"# } else { "end" };
        let _ = s.tx.send(SttMsg::text(end)).await;
        // 엔진이 남은 결과를 보내고 연결을 닫을 때까지 대기. 파일 업로드는 실시간보다 빨리 보내므로
        // 밀린 처리량이 오디오 길이만큼 있을 수 있어 상한을 오디오 길이에 비례시킨다.
        let wait = 10 + (total / RATE as usize) as u64;
        let _ = tokio::time::timeout(Duration::from_secs(wait), async {
            while let Some(Ok(m)) = s.rx.next().await {
                if let SttMsg::Text(t) = m
                    && let Ok(true) = s.apply(&mut tr, t.as_str())
                {
                    let _ = sock.send(Message::Text(tr.to_json().into())).await;
                }
            }
        })
        .await;
    }
    if let Some((url, pcm)) = batch {
        let secs = pcm.len() / RATE as usize;
        sock.send(status(&format!("변환 중... (녹음 {:02}:{:02})", secs / 60, secs % 60))).await?;
        match transcribe_batch(&url, &pcm).await.and_then(|r| tr.apply_segments(&r)) {
            Ok(_) => sock.send(Message::Text(tr.to_json().into())).await?,
            Err(e) => {
                println!("[{path}] 변환 실패: {e}");
                sock.send(status(&format!("변환 실패: {e}"))).await?
            }
        }
    }
    println!("[{path}] 완료: 발언 {}개", tr.finals.len());
    let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    let saved = json!({ "type": "saved", "path": path, "seconds": secs, "bytes": bytes });
    sock.send(Message::Text(saved.to_string().into())).await?;
    Ok(())
}
