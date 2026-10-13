"""Website version. Run locally:  streamlit run app.py"""
import tempfile
from pathlib import Path

import streamlit as st

from clipper import audio_energy, candidates, cut, download, pick, transcribe

st.set_page_config(page_title="YouTube Auto-Clipper", page_icon="🎬")
st.title("🎬 YouTube Auto-Clipper")
st.caption("Paste a link (or upload a video) and get the best 30-60s clips. Free.")

with st.sidebar:
    st.header("Settings")
    num = st.slider("Number of clips", 1, 10, 5)
    min_len, max_len = st.slider("Clip length (seconds)", 15, 90, (30, 60))
    model = st.selectbox("Transcription model", ["tiny", "base", "small"], index=1,
                         help="Bigger = more accurate but slower")
    vertical = st.checkbox("Vertical 9:16 (Shorts/TikTok)")

mode = st.radio("Source", ["YouTube link", "Upload a video file"], horizontal=True)
url = st.text_input("YouTube URL") if mode == "YouTube link" else None
upload = st.file_uploader("Video file", type=["mp4", "mov", "mkv", "webm"]) if mode != "YouTube link" else None

if st.button("Find best clips", type="primary"):
    if not (url or upload):
        st.error("Add a link or upload a file first.")
    else:
        st.session_state.pop("clips", None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                with st.status("Working...", expanded=True) as status:
                    if url:
                        st.write("Downloading video...")
                        video, _ = download(url, tmp)
                    else:
                        video = tmp / ("upload" + Path(upload.name).suffix)
                        video.write_bytes(upload.getbuffer())
                    st.write("Analysing audio...")
                    energy = audio_energy(video)
                    st.write("Transcribing (first run downloads the model)...")
                    segs = transcribe(video, model)
                    best = pick(candidates(segs, energy, min_len, max_len), num)
                    results = []
                    for k, c in enumerate(best, 1):
                        st.write(f"Cutting clip {k}/{len(best)}...")
                        f = tmp / f"clip_{k}.mp4"
                        cut(video, c, f, vertical)
                        results.append((c, f.read_bytes()))
                    status.update(label="Done!", state="complete")
                st.session_state["clips"] = results
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            if url:
                st.info("YouTube often blocks downloads from cloud servers. "
                        "Try running the app locally, or use the 'Upload a video file' option.")

for k, (c, data) in enumerate(st.session_state.get("clips", []), 1):
    st.subheader(f"Clip {k}  ·  {int(c['start'])}s - {int(c['end'])}s  ·  score {c['score']:.2f}")
    st.video(data)
    st.caption(c["text"][:200] + "...")
    st.download_button("Download", data, file_name=f"clip_{k}.mp4", mime="video/mp4", key=f"dl{k}")
