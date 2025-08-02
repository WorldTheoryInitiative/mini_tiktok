# tiktok_style.py
import streamlit as st
import requests
import json
import numpy as np

API_URL = "http://localhost:8000"
URL_FILE = "video_urls.json"

@st.cache_data
def load_urls():
    with open(URL_FILE, "r") as f:
        return json.load(f)

urls = load_urls()

@st.cache_data
def load_hashtags():
    with open("../data/trending_videos.json", "r", encoding="utf-8") as f:
        raw = json.load(f)["collector"]
    ht = []
    for v in raw:
        names = [f"#{h['name']}" for h in v.get("hashtags", [])]
        ht.append(" ".join(names[:3]))
    return ht

hashtags = load_hashtags()

# ---------- session ----------
if "ids" not in st.session_state:
    st.session_state.ids = []
if "idx" not in st.session_state:
    st.session_state.idx = 0
if "user_id" not in st.session_state:
    st.session_state.user_id = 42

# User picker + refresh button
new_user = st.number_input("User ID", 0, 499, st.session_state.user_id)
if st.button("Load / Refresh Feed"):
    st.session_state.user_id = new_user
    st.session_state.ids = requests.get(
        f"{API_URL}/recommend", params={"user_id": new_user}
    ).json()
    st.session_state.idx = 0
    st.rerun()

if not st.session_state.ids:
    st.stop()

# ---------- current video ----------
rec = st.session_state.ids[st.session_state.idx]
post_id = urls[rec["video_id"]].split("/")[-1]
embed = f"https://www.tiktok.com/player/v1/{post_id}?controls=1&autoplay=0&music_info=1&description=1"

col_video, col_stats = st.columns([2, 1])
with col_video:
    st.components.v1.iframe(embed, width=340, height=600, scrolling=False)

with col_stats:
    st.metric("Similarity", f"{rec['score']:.2f}")
    st.metric("Percentile", f"{rec['percentile']:.0f} %")
    st.write("Top tags:", rec["hashtags"] if rec["hashtags"] else "—")

# ---------- navigation ----------
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("⬅ Prev") and st.session_state.idx > 0:
        st.session_state.idx -= 1
        st.rerun()
with col2:
    if st.button("➡ Next"):
        st.session_state.idx = (st.session_state.idx + 1) % len(st.session_state.ids)
        st.rerun()
with col3:
    if st.button("👍"):
        requests.post(API_URL + "/interact", json={
            "user_id": st.session_state.user_id,
            "video_id": rec["video_id"],
            "action": "like"
        })
with col4:
    if st.button("👎"):
        requests.post(API_URL + "/interact", json={
            "user_id": st.session_state.user_id,
            "video_id": rec["video_id"],
            "action": "dislike"
        })