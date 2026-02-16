# Gemini Implicit Caching Dead Zone

This script tests a weird caching issue with **`gemini-3-flash-preview`** where implicit caching just stops working between ~9K–17K prompt tokens. Like, it drops to **0** even though we're sending the exact same prefix every time. Then after that, it locks to weird plateaus (~16K, ~24K, ~32K) instead of growing smoothly.

## What's happening

The script makes a conversation that grows with each turn. Same system prompt every time, same conversation history - just adding new stuff to the end. Nothing changes in what I have sent before.

The test runs from 1K to 35K tokens (going up by 1K each time) and check how much the API says it cached from the response.

**What should happen:** The cache should grow as we send more of the same stuff, or at worst stay the same. It definitely shouldn't drop to 0.

**What actually happens:** Three weird phases:

| Regime | Prompt range | Behavior |
|--------|-------------|----------|
| Normal caching | 1K – ~8K | Cache grows in ~2,048-token blocks (2K → 4K → 6K) although gemini docs says it should grow by 1024 tokens |
| **Dead zone** | **~9K – ~17K** | **`cached_content_token_count` = 0** despite stable prefix |
| Plateau locking | ~18K – 35K+ | Cache snaps to fixed levels (~16K, ~24K, ~32K) and stays locked for ~8 turns each |

## The script would produce an output like this

```
Turn | Target | Prompt  | Cached  |    New | Cache% |  Time | Notes
----------------------------------------------------------------------
   1 |  1,000 |   1,001 |       0 |  1,001 |  0.0%  | 1.23s |
   2 |  2,000 |   2,013 |       0 |  2,013 |  0.0%  | 0.98s |
   3 |  3,000 |   3,038 |   2,176 |    862 | 71.6%  | 0.87s | +2,176 cached (≈1 blk)
   4 |  4,000 |   4,029 |   2,176 |  1,853 | 54.0%  | 0.91s |
   5 |  5,000 |   5,013 |   4,224 |    789 | 84.3%  | 0.84s | +4,224 cached (≈2 blk)
   6 |  6,000 |   6,025 |   4,224 |  1,801 | 70.1%  | 0.90s |
   7 |  7,000 |   7,028 |   6,272 |    756 | 89.2%  | 0.82s | +6,272 cached (≈3 blk)
   8 |  8,000 |   8,012 |   6,272 |  1,740 | 78.3%  | 0.88s |
   9 |  9,000 |   9,032 |       0 |  9,032 |  0.0%  | 1.10s | ← DROPPED TO 0
  10 | 10,000 |  10,041 |       0 | 10,041 |  0.0%  | 1.15s |
  ...
  17 | 17,000 |  17,008 |       0 | 17,008 |  0.0%  | 1.20s |
  18 | 18,000 |  18,042 |  16,279 |  1,763 | 90.2%  | 0.95s | +16,279 cached (≈7 blk)
  19 | 19,000 |  19,021 |  16,279 |  2,742 | 85.6%  | 0.97s |
  ...
  26 | 26,000 |  26,038 |  24,407 |  1,631 | 93.7%  | 0.88s | +24,407 cached (≈11 blk)
  ...
  34 | 34,000 |  34,019 |  32,487 |  1,532 | 95.5%  | 0.85s | +32,487 cached (≈15 blk)
```

## Setup

```bash
uv sync
uv run main.py --api_key YOUR_KEY
```
### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--api_key` | `.env` | Gemini API key |
| `--start` | `1000` | Starting token count |
| `--stop` | `35000` | Ending token count |
| `--step` | `1000` | How much to increase each turn |
| `--sleep` | `2.0` | Seconds to wait between API calls for cache to persist|
| `--max_tokens` | `16` | Max tokens the model can output, this help to reduce noise in the output while varying the user prompt |


## How it works

1. Sets a system prompt (~405 chars) that stays the same every call
2. Sends the full conversation history each time (user message, model response, repeat) plus a new question
3. Adds padding to hit the next 1K token target
4. Asks the model to just say "OK" (with `max_output_tokens=16` and `temperature=0`) to keep things consistent
5. Grabs the `cached_content_token_count` from the response and logs it

