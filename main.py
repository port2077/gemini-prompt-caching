
import argparse
import os
import time
from pathlib import Path

from google import genai
from google.genai import types

MODEL = "gemini-3-flash-preview"

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about world geography, "
    "capital cities, population statistics, and general knowledge. You provide "
    "accurate, factual information. When asked to reply with one word, comply "
    "exactly. You have deep knowledge of countries, capitals, major cities, "
    "rivers, mountain ranges, climate zones, and economic indicators. Always be "
    "concise and precise in your responses."
)

# Adaptive padding constants (empirical for this model)
TOKENS_PER_CHAR = 0.19
BASELINE_OVERHEAD = 240
TURN_OVERHEAD = 26
RESPONSE_TOKENS = 8


def load_env():
    for p in [Path.cwd() / ".env", Path(__file__).resolve().parent / ".env"]:
        if p.exists():
            for line in p.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip("\"'")
                    if k and k not in os.environ:
                        os.environ[k] = v
            break


def make_padding(turn, n_chars):
    base = (
        f"[Turn {turn}] Geography reference: countries, capitals, populations, "
        f"area, GDP, climate, languages, currency, time zones, coordinates. "
    )
    if n_chars <= 0:
        return ""
    return (base * (n_chars // len(base) + 1))[:n_chars]


def make_question(turn, padding_chars=0):
    q = f"Question {turn}: what is the capital of country number {turn}?"
    if padding_chars <= 0:
        return f"{q} Reply with one word: OK"
    return f"{q}\n\nReference:\n{make_padding(turn, padding_chars)}\n\nReply with one word: OK"


def extract_usage(resp):
    u = getattr(resp, "usage_metadata", None)
    if u is None:
        return 0, 0, 0
    d = u.model_dump() if hasattr(u, "model_dump") else (
        u.to_dict() if hasattr(u, "to_dict") else u.__dict__
    )
    def gi(keys):
        for k in keys:
            v = d.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return int(v)
        return 0
    return (
        gi(["prompt_token_count", "promptTokenCount"]),
        gi(["candidates_token_count", "candidatesTokenCount"]),
        gi(["cached_content_token_count", "cachedContentTokenCount"]),
    )


def main():
    ap = argparse.ArgumentParser(
        description="Reproduce Gemini implicit caching dead zone (~9K-17K)")
    ap.add_argument("--api_key", default=None)
    ap.add_argument("--start", type=int, default=1000)
    ap.add_argument("--stop", type=int, default=35000)
    ap.add_argument("--step", type=int, default=1000)
    ap.add_argument("--sleep", type=float, default=2.0)
    ap.add_argument("--max_tokens", type=int, default=16)
    args = ap.parse_args()

    load_env()
    api_key = args.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY in .env or pass --api_key")

    targets = list(range(args.start, args.stop + 1, args.step))
    client = genai.Client(api_key=api_key)

    print(f"Model: {MODEL}")
    print(f"System prompt: {len(SYSTEM_PROMPT)} chars (fixed across all turns)")
    print(f"Sweep: {args.start:,} → {args.stop:,} prompt tokens, step={args.step:,}")
    print(f"Turns: {len(targets)}, sleep: {args.sleep}s between calls")
    print(f"google-genai version: {genai.__version__}")
    print()
    print(f"{'Turn':>4} | {'Target':>6} | {'Prompt':>7} | {'Cached':>7} | "
          f"{'New':>7} | {'Cache%':>6} | {'Time':>6} | Notes")
    print("-" * 95)

    pad_plan = [0] * len(targets)
    pad_plan[0] = max(0, int((targets[0] - BASELINE_OVERHEAD) / TOKENS_PER_CHAR))

    history = []
    tpc, exp_resp = TOKENS_PER_CHAR, RESPONSE_TOKENS
    prev_prompt, prev_hchars = None, 0
    prev_qlen = len(make_question(1, pad_plan[0]))
    prev_cached = 0

    for i, target in enumerate(targets):
        turn = i + 1
        question = make_question(turn, pad_plan[i])

        # Build multi-turn contents
        contents = []
        for h in history:
            contents.append(types.UserContent(
                parts=[types.Part.from_text(text=h["q"])]))
            contents.append(types.ModelContent(
                parts=[types.Part.from_text(text=h["a"])]))
        contents.append(types.UserContent(
            parts=[types.Part.from_text(text=question)]))

        t0 = time.time()
        resp = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0,
                max_output_tokens=args.max_tokens,
            ),
        )
        elapsed = time.time() - t0

        prompt, compl, cached = extract_usage(resp)
        ratio = cached / max(prompt, 1)

        note = ""
        if cached > prev_cached:
            note = f"+{cached - prev_cached:,} cached (≈{cached // 2048} blk)"
        elif cached == 0 and prev_cached > 0:
            note = "← DROPPED TO 0"
        prev_cached = cached

        print(f"{turn:>4} | {target:>6,} | {prompt:>7,} | {cached:>7,} | "
              f"{prompt - cached:>7,} | {ratio:>5.1%} | {elapsed:>5.2f}s | {note}")

        txt = getattr(resp, "text", "") or ""
        if not txt and hasattr(resp, "candidates") and resp.candidates:
            try:
                txt = resp.candidates[0].content.parts[0].text
            except Exception:
                txt = ""
        history.append({"q": question, "a": str(txt)})

        # Adaptive padding calibration
        hchars = sum(len(h["q"]) + len(h["a"]) for h in history)
        if prev_prompt is not None:
            dt = prompt - prev_prompt
            dc = (hchars - prev_hchars) + (len(question) - prev_qlen)
            if dt > 0 and dc > 0:
                tpc = min(0.6, max(0.05, 0.75 * tpc + 0.25 * (dt / dc)))
        if compl > 0:
            exp_resp = 0.7 * exp_resp + 0.3 * compl
        if i + 1 < len(targets):
            base_q = len(make_question(turn + 1, 0))
            needed = targets[i + 1] - prompt - exp_resp - base_q * tpc - TURN_OVERHEAD
            pad_plan[i + 1] = max(0, min(int(needed / tpc), 50000))

        prev_prompt, prev_hchars, prev_qlen = prompt, hchars, len(question)
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
