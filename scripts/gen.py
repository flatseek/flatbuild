#!/usr/bin/env python3

import json
import time
from pathlib import Path

import requests
from tqdm import tqdm

API_URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL = "local-model"

OUTPUT = Path("dataset.jsonl")

TOTAL = 10000

SYSTEM = """
You are Flatseek Assistant.

Generate synthetic training conversations.

Return ONLY one JSON object.

Format:

{
  "messages":[
    {
      "role":"system",
      "content":"You are a helpful, friendly, and concise conversational assistant."
    },
    {
      "role":"user",
      "content":"..."
    },
    {
      "role":"assistant",
      "content":"..."
    }
  ],
  "metadata":{
    "generator":"synthetic"
  }
}

Rules:

- JSON only.
- No markdown.
- No explanations.
- Natural conversation.
- Different every time.
- 2-6 turns.
- Mix short and long answers.
- Everyday topics.
- Friendly.
"""

existing = 0

if OUTPUT.exists():
    with OUTPUT.open() as f:
        existing = sum(1 for _ in f)

print(f"Existing : {existing}")
print(f"Target   : {TOTAL}")

bar = tqdm(total=TOTAL, initial=existing)

with OUTPUT.open("a", encoding="utf8") as out:

    while existing < TOTAL:

        payload = {
            "model": MODEL,
            "temperature": 0.9,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM,
                },
                {
                    "role": "user",
                    "content": "Generate one unique conversation.",
                },
            ],
        }

        try:

            r = requests.post(
                API_URL,
                json=payload,
                timeout=300,
            )

            r.raise_for_status()

            text = r.json()["choices"][0]["message"]["content"].strip()

            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                text = text.rsplit("```", 1)[0]

            obj = json.loads(text)

            out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            out.flush()

            existing += 1
            bar.update(1)

        except KeyboardInterrupt:
            break

        except Exception as e:
            print(e)
            time.sleep(2)

bar.close()

print("Done.")