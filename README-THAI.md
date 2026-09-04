# ผู้ช่วยเสียงภาษาไทย + RAG

ต่อยอดจาก Granite-chan เดิม เปลี่ยนทั้ง pipeline ให้รองรับภาษาไทย

```
เสียงพูด (ไทย)
   ↓  Typhoon Whisper Turbo         app/stt.py
ข้อความไทย
   ↓  BGE-M3 + FAISS                app/rag.py
ข้อมูลอ้างอิงจากเอกสาร
   ↓  Typhoon LLM (OpenTyphoon API) app/llm.py
คำตอบภาษาไทย
   ↓  edge-tts th-TH               app/tts.py
เสียงพูดกลับ
```

## เปลี่ยนอะไรจากของเดิมบ้าง

| ส่วน | เดิม | ใหม่ | เหตุผล |
|---|---|---|---|
| STT | faster-whisper `language='en'` | `typhoon-ai/typhoon-whisper-turbo` | fine-tune บนเสียงไทย ~11,000 ชม. |
| Embeddings | IBM SLATE 125M **ENG** | `BAAI/bge-m3` | ตัวเดิมเป็นอังกฤษล้วน retrieval ภาษาไทยพังเงียบ ๆ |
| LLM | IBM Granite 3 8B | `typhoon-v2.5-30b-a3b-instruct` | Granite ภาษาไทยอ่อน |
| TTS | Piper `en_GB` + `piper.exe` | edge-tts `th-TH` | Piper ไม่มีเสียงไทย และ `.exe` รันบน macOS ไม่ได้ |
| Framework | LangChain 0.1 (API ถูกลบแล้ว) | เขียนตรง ไม่มี LangChain | `ConversationalRetrievalChain`, `langchain.vectorstores` ถูกถอดออกไปแล้ว |
| Interface | CLI loop | FastAPI service | ต่อ UI ทีหลังได้ |

> **สำคัญ:** API key ของ IBM watsonx ถูก hardcode ไว้ที่ [advanceRAG_api.py:28](advanceRAG_api.py:28) และ [:154](advanceRAG_api.py:154) และอยู่ใน git history แล้ว **ควรไป revoke ทิ้งที่ IBM Cloud console** ไม่ว่าจะยังใช้ watsonx ต่อหรือไม่ก็ตาม

โค้ดเดิม (`main.py`, `advanceRAG_api.py`, `piper/`) ยังอยู่ครบ ไม่ได้ลบ ใช้อ้างอิงได้

---

## ติดตั้ง

ต้องใช้ venv **แยกจากของเดิม** เพราะ `requirements.txt` เก่าตรึง LangChain รุ่นที่เลิกใช้แล้ว

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements-thai.txt
```

ถ้าจะรับไฟล์เสียงจากเบราว์เซอร์ (webm/opus) ต้องมี ffmpeg ด้วย:

```bash
brew install ffmpeg
```

### ตั้งค่า

```bash
cp .env.example .env
```

แล้วใส่ `TYPHOON_API_KEY` (ขอฟรีที่ https://playground.opentyphoon.ai → API Keys)

### พื้นที่ดิสก์

ครั้งแรกจะโหลดโมเดลลง `~/.cache/huggingface` ประมาณ **6 GB**
- `typhoon-whisper-turbo` ~1.6 GB
- `bge-m3` ~4.3 GB

---

## ใช้งาน

### 1. สร้าง index จากเอกสาร

วางไฟล์ `.pdf` / `.txt` / `.md` ไว้ใน `data/` แล้ว

```bash
python -m scripts.ingest
```

รันใหม่ทุกครั้งที่เพิ่มหรือแก้เอกสาร (index เก็บที่ `storage/`)

### 2. ทดสอบด้วย terminal ก่อน

```bash
python -m scripts.ask
```

พิมพ์คำถามไทยได้เลย เพิ่ม `--speak` เพื่อให้อ่านออกเสียงด้วย หรือ `--audio clip.wav` เพื่อทดสอบจากไฟล์เสียง

### 3. รัน API

```bash
uvicorn app.server:app --reload --port 8000
```

เปิด http://localhost:8000/docs เพื่อดู Swagger UI

---

## Endpoints

| Method | Path | รับ | คืน |
|---|---|---|---|
| GET | `/health` | – | โมเดลที่กำลังใช้อยู่ |
| POST | `/transcribe` | ไฟล์เสียง | `{text, language, duration}` |
| POST | `/chat` | `{message, session_id}` | `{answer, sources}` |
| POST | `/chat/stream` | `{message, session_id}` | SSE ทีละ token |
| POST | `/speak` | `{text, voice}` | `audio/mpeg` |
| POST | `/voice` | ไฟล์เสียง | `{transcript, answer, sources, audio_base64}` |
| POST | `/session/reset` | `?session_id=` | – |
| GET | `/voices` | – | เสียงไทยที่ใช้ได้ |

ตัวอย่าง — ครบวงจรในคำสั่งเดียว:

```bash
curl -s -X POST "http://localhost:8000/voice?session_id=u1" -F "file=@question.wav" | jq '{transcript, answer}'
```

ถอดเสียงอย่างเดียว:

```bash
curl -s -X POST http://localhost:8000/transcribe -F "file=@question.wav"
```

ถามด้วยข้อความ:

```bash
curl -s -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"message":"โซนเสื้อผ้าอยู่ชั้นไหน","session_id":"u1"}'
```

---

## จูนคุณภาพ

**ตอบว่า "ไม่มีข้อมูล" บ่อยเกินไป** → ลด `RAG_MIN_SCORE` (0.35 → 0.25) หรือเพิ่ม `RAG_TOP_K`

**ตอบมั่ว / หลุดจากเอกสาร** → เพิ่ม `RAG_MIN_SCORE` และลด `LLM_TEMPERATURE`

**คำตอบโดนตัดกลางประโยค** → เพิ่ม `LLM_MAX_TOKENS` (ภาษาไทยกินโทเคนราว 2-3 เท่าของอังกฤษ ของเดิมตั้งไว้ 69 จึงตัดตลอด)

**ค้นไม่เจอทั้งที่มีข้อมูล** → ปรับ `CHUNK_SIZE` ถ้าเอกสารเป็นตาราง/รายการสั้น ๆ ให้ลดเหลือ ~400

**STT ช้าบน CPU** → แปลงโมเดลเป็น CTranslate2 แล้วสลับ backend

```bash
ct2-transformers-converter --model typhoon-ai/typhoon-whisper-turbo --output_dir models/typhoon-whisper-turbo-ct2 --quantization int8
```

แล้วตั้ง `STT_BACKEND=faster-whisper` และ `STT_MODEL=models/typhoon-whisper-turbo-ct2`

**อยากได้ latency ต่ำแบบ streaming จริง** → `typhoon-asr-realtime` (FastConformer) เร็วกว่ามากบน CPU แต่ต้องเปลี่ยน audio pipeline เป็น NeMo ทั้งชุด

---

## หมายเหตุก่อนขึ้น production

- CORS ตั้ง `allow_origins=["*"]` ไว้สำหรับ dev — ต้องแก้ก่อนเปิดออกนอก localhost ([app/server.py](app/server.py))
- session history เก็บใน memory ของ process เดียว ถ้าจะ scale หลาย worker ต้องย้ายไป Redis
- ยังไม่มี auth บน endpoint
