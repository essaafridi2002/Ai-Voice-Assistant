import streamlit as st
import requests
import time
import os
from speech_to_text import record_audio_realtime
from text_to_speech import speak_fast

st.set_page_config(page_title="ARIA · AI Assistant", page_icon="◈", layout="centered",
                   initial_sidebar_state="collapsed")

# ── Session state ─────────────────────────────────────────────────────────────
if "messages"     not in st.session_state: st.session_state.messages     = []
if "last_timing"  not in st.session_state: st.session_state.last_timing  = {}
if "is_recording" not in st.session_state: st.session_state.is_recording = False

# ── LLM with memory ───────────────────────────────────────────────────────────
def ask_llama(user_msg: str, history: list, mode: str = "chat") -> tuple[str, float]:
    if mode == "voice":
        system = ("You are ARIA, a fast AI voice assistant. "
                  "Reply in 1-2 short sentences. No markdown. Natural conversational speech.")
    else:
        system = ("You are ARIA, a fast AI assistant. "
                  "Be concise: 1-3 sentences max. Plain text only. No lists or bullet points.")

    # Memory: last 6 exchanges (12 messages)
    recent = history[-12:]
    convo  = "".join(
        f"{'User' if m['role']=='user' else 'ARIA'}: {m['content']}\n"
        for m in recent
    )
    full_prompt = f"{system}\n\n{convo}User: {user_msg}\nARIA:"

    t0 = time.time()
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.6,
                    "num_predict": 80,   # ← reduced from 120 → faster
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                    "num_ctx": 1024,     # ← smaller context window → faster
                },
            },
            timeout=60,
        )
        ms = (time.time() - t0) * 1000
        return r.json()["response"].strip(), ms
    except requests.exceptions.ConnectionError:
        return "⚠️ Ollama not running. Start it with: ollama serve", 0
    except Exception as e:
        return f"⚠️ Error: {e}", 0

# ── Message renderer ──────────────────────────────────────────────────────────
def render_msg(msg: dict):
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        via = msg.get("via", "text")
        icon = "🎤" if via == "voice" else "⌨"
        ms = f"{msg['ms']:.0f}ms" if msg.get("ms") else ""
        if msg["role"] == "assistant" and "audio_path" in msg:
            if st.button("🔊", key=f"play_{hash(msg['content'])}"):
                st.session_state.current_audio = msg["audio_path"]
        st.caption(f"{icon} {via} {ms}")

# ── Layout ────────────────────────────────────────────────────────────────────
with st.sidebar:
    total     = len(st.session_state.messages)
    voice_cnt = sum(1 for m in st.session_state.messages if m.get("via") == "voice")
    usr_cnt   = sum(1 for m in st.session_state.messages if m["role"] == "user")

    st.markdown(f"""
    <div class="side-panel">
      <div class="logo"><div class="logo-dot"></div>ARIA</div>
      <div class="s-label">Model</div>
      <div class="badge"><div class="badge-dot"></div>llama3.2 · local · Ollama</div>
      <div class="s-label">Session</div>
      <div class="stat"><div class="stat-val">{total}</div><div class="stat-lbl">messages total</div></div>
      <div class="stat"><div class="stat-val">{voice_cnt}</div><div class="stat-lbl">voice turns</div></div>
      <div class="stat"><div class="stat-val">{min(usr_cnt,6)}/6</div><div class="stat-lbl">memory window</div></div>
      <div class="s-label">Pipeline</div>
      <div class="tip">
        STT · faster-whisper tiny<br>
        LLM · llama3.2 (Ollama)<br>
        TTS · edge-tts GuyNeural<br>
        Memory · last 6 exchanges
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    if st.button("🗑 Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_timing = {}
        st.rerun()

st.markdown("""
<div style="padding:4px 0 16px;border-bottom:1px solid #111820;margin-bottom:20px;
     display:flex;align-items:center;justify-content:space-between;">
  <span style="font-family:'Syne',sans-serif;font-size:.9rem;font-weight:600;color:#4a6a90;
        letter-spacing:.06em;">ARIA · Adaptive Response Intelligence Assistant</span>
  <span style="font-size:.68rem;color:#1e3050;background:#0d1520;border:1px solid #1a2535;
        padding:3px 10px;border-radius:20px;">◈ Voice + Chat + Memory</span>
</div>
""", unsafe_allow_html=True)

# Chat history
if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
      <h2>◈</h2>
      <p>Type a message or press <strong style="color:#4a6a90">🎤 Speak</strong> to start.<br>
      ARIA remembers your last 6 exchanges automatically.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        render_msg(msg)

# Input zone
st.markdown('<div class="input-zone">', unsafe_allow_html=True)

if st.session_state.is_recording:
    st.markdown("""
    <div class="rec-banner">
      <div class="rec-dot"></div>
      🎤 Recording… speak now (stops automatically on silence)
    </div>""", unsafe_allow_html=True)

c_txt, c_send, c_voice = st.columns([6, 1, 1])
with c_txt:
    user_text = st.text_input("Message", placeholder="Message ARIA…",
                              label_visibility="collapsed", key="chat_inp")
with c_send:
    send = st.button("Send →", use_container_width=True)
with c_voice:
    st.markdown('<div class="voice-col">', unsafe_allow_html=True)
    voice = st.button("🎤 Speak", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Timing chips
t = st.session_state.last_timing
if t:
    chips = ""
    if t.get("llm"): chips += f'<span class="tchip">LLM {t["llm"]:.0f}ms</span>'
    if t.get("tts"): chips += f'<span class="tchip">TTS {t["tts"]:.0f}ms</span>'
    if t.get("stt"): chips += f'<span class="tchip">STT {t["stt"]:.0f}ms</span>'
    st.markdown(f'<div class="timing">⚡ {chips}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── Text handler ──────────────────────────────────────────────────────────
if send and user_text.strip():
    st.session_state.messages.append({"role":"user","content":user_text.strip(),"via":"text"})
    with st.spinner("◈ Thinking…"):
        reply, llm_ms = ask_llama(user_text.strip(), st.session_state.messages, "chat")
    t_tts = time.time()
    audio_path = f"response_{len(st.session_state.messages) + 1}.mp3"
    speak_fast(reply, audio_path)
    tts_ms = (time.time() - t_tts) * 1000
    st.session_state.messages.append({"role":"assistant","content":reply,"via":"text","ms":llm_ms, "audio_path": audio_path})
    st.session_state.last_timing = {"llm": llm_ms, "tts": tts_ms}

# ── Voice handler ─────────────────────────────────────────────────────────
if voice:
    st.session_state.is_recording = True
    st.rerun()

if st.session_state.is_recording:
    st.session_state.is_recording = False
    with st.spinner("🎤 Listening… (stops on silence)"):
        t0 = time.time()
        transcript = record_audio_realtime()
        stt_ms = (time.time() - t0) * 1000

    if not transcript.strip():
        st.warning("No speech detected. Please try again.")
    else:
        st.session_state.messages.append({"role":"user","content":transcript.strip(),"via":"voice"})
        with st.spinner("◈ Thinking…"):
            reply, llm_ms = ask_llama(transcript.strip(), st.session_state.messages, "voice")
        with st.spinner("🔊 Speaking…"):
            t2 = time.time()
            audio_path = f"response_{len(st.session_state.messages) + 1}.mp3"
            speak_fast(reply, audio_path)
            tts_ms = (time.time() - t2) * 1000
        st.session_state.messages.append({"role":"assistant","content":reply,"via":"voice","ms":llm_ms, "audio_path": audio_path})
        st.session_state.last_timing = {"llm": llm_ms, "tts": tts_ms, "stt": stt_ms}

    st.rerun()

if "current_audio" in st.session_state:
    st.audio(st.session_state.current_audio, autoplay=False)