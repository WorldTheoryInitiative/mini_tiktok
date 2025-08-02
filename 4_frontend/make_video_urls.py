#!/usr/bin/env python3
import json
import pandas as pd

# 1. load the source JSON
with open("data/trending_videos.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 2. grab the 'collector' list (our 1000 videos)
videos = data["collector"]

# 3. pull the clean TikTok share URL from each entry
urls = [v["webVideoUrl"].split('">')[0] for v in videos]

# 4. optional: keep only the 1000 we actually kept
urls = urls[:1000]

# 5. save as a flat JSON list
with open("4_frontend/src/video_urls.json", "w", encoding="utf-8") as f:
    json.dump(urls, f, indent=2)

print(f"✅ wrote {len(urls)} URLs → frontend/src/video_urls.json")