# paperbot, articlebot & zoterobot

Three Slack/Zulip bots for a research group, running fully on a local GPU server with [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server` behind [llama-swap](https://github.com/mostlygeek/llama-swap) — no external LLM API.

| Bot | What it does | When it runs | GPU |
|---|---|---|---|
| **paperbot** | Finds new papers on Semantic Scholar and arXiv, filters them by your interests, summarizes each one with a local LLM, and posts to Slack channels | Once a day (tmux runner, default 07:00) | During the run only |
| **articlebot** | Collects posts from company research blogs (Google Research, DeepMind, NVIDIA, Meta …) via RSS, scores them against your interests, keeps a backlog, and posts one summarized post per organisation per day | Right after paperbot, in the same runner | During the run only |
| **zoterobot** | Saves a posted paper to your Zotero library + Google Drive when you mention it in a thread, and answers DMs about your library (a tool-calling chat agent) | Always on (Socket Mode listener) | Only while answering a DM |

All three live in this repo as separate packages (`paperbot/`, `articlebot/`, `zoterobot/`) and share one virtualenv, one local model (served through llama-swap), and one Zulip bot account. articlebot and zoterobot reuse paperbot's helpers (llama-swap control, Zulip client and message layout) and zoterobot's full-text cache instead of carrying copies. On Slack, paperbot and articlebot post through one app and zoterobot is a second app with its own tokens.

---

## Contents

1. [How it works](#1-how-it-works)
2. [Repository layout](#2-repository-layout)
3. [Setup](#3-setup)
4. [paperbot](#4-paperbot)
5. [zoterobot](#5-zoterobot)
6. [articlebot](#6-articlebot)
7. [Operations notes](#7-operations-notes)
8. [Code map](#8-code-map)

---

## 1. How it works

### 1.1 Overview

```
~/paperbot   (one venv · one LLM via llama-swap :8090 · one Zulip bot)
├── paperbot     daily 07:00            papers  → summary → post
├── articlebot   right after paperbot   blog posts → summary → post
└── zoterobot    always on              save command + DM chat agent
```

Shared store: the chat cache (`library_text.db`). paperbot and articlebot write every published item into it; zoterobot also fills it from Zotero/Drive and searches it.

### 1.2 Daily runner — `scripts/loop.sh` (tmux `paperbot`)

```
07:00 → GPUs 1,2 have ≥ 13 GB free? ── no → re-check every 30 min, skip the day after 2 h
           │ yes
           ├─ restart llama-swap if it does not respond
           ├─ python -m paperbot
           └─ scripts/articlebot.sh        (3 h limit; only if articlebot/config.yaml exists)
```

### 1.3 paperbot

| Step | What happens |
|---|---|
| 1. Fetch | Per topic: Semantic Scholar + arXiv; papers already in the DB are skipped |
| 2. Filter | Embedding similarity → LLM judge (0–10); below threshold → `filtered_out` |
| 3. Quota | `pending` papers first (FIFO), then new ones, up to the daily cap per topic/source; the rest → `pending` |
| 4. Process | PDF → text + figures/tables → six-section summary |
| 5. Publish | Slack or Zulip (per topic `platform`) → chat cache |
| 6. Finish | Unload the model (free the GPU) |

Temporary: Zulip topics first replay older papers (3 per day) and hold new ones until the replay is done (`paperbot/replay.py`).

### 1.4 articlebot

| Step | When | What happens |
|---|---|---|
| 1. Ingest | every 3 days | RSS → LLM score from the first 3,000 chars of the body → score ≥ 5 goes to the backlog (DB) |
| 2. Report | daily | Per organisation: newest backlog post → body + images → summary → Slack/Zulip → chat cache |
| 3. Finish | each run | Unload the model unless paperbot is running |

### 1.5 zoterobot (tmux `zoterobot`)

| Input | What happens |
|---|---|
| Slack thread mention / Zulip DM `저장 <collection>` | Identify the paper → Zotero item + PDF on Google Drive (linked file) |
| Any other DM | The LLM calls tools (search / read / compare / summarize) over the chat cache and answers from the full text |

### 1.6 Shared code

```
paperbot.llmserver        ← articlebot, zoterobot   (model loaded? / unload after in-flight requests)
paperbot.publisher.zulip  ← articlebot, zoterobot   (Zulip client, message layout, math spacing)
zoterobot.textcache       ← paperbot, articlebot    (writing into the chat cache)
```

---

## 2. Repository layout

```
paperbot/          paper bot package          python -m paperbot     config: config.yaml
articlebot/        research-blog bot package  python -m articlebot   config: articlebot/config.yaml
zoterobot/         Zotero + chat bot package  python -m zoterobot    config: zoterobot/config.yaml
scripts/           tmux runners (loop.sh, articlebot.sh, zoterobot.sh)
logs/              runtime logs (gitignored)
requirements.txt   one dependency list for all three bots
```

Every Python module is listed with its role in [8. Code map](#8-code-map).

---

## 3. Setup

### 3.1 Python

Python 3.10+. Install the CPU builds of torch and torchvision first so the embedding model and the layout detector do not pull the CUDA wheels (a CUDA torchvision next to a CPU torch fails with `operator torchvision::nms does not exist`).

```bash
cd ~/paperbot
python3 -m venv venv && source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cp config.example.yaml config.yaml
cp articlebot/config.example.yaml articlebot/config.yaml   # optional — without it loop.sh skips articlebot
cp zoterobot/config.example.yaml zoterobot/config.yaml
```

### 3.2 LLM server: llama.cpp + llama-swap

The bots talk to [llama-swap](https://github.com/mostlygeek/llama-swap) on `127.0.0.1:8090` (OpenAI-compatible `/v1`). llama-swap starts `llama-server` when the first request arrives and stops it `ttl` seconds after the last one, so the GPUs are only held while a bot is working. Everything runs as a normal user (no sudo). On this server (V100 ×3, NVIDIA driver 535) a 52k-token paper takes about 2 minutes to summarize.

**Build llama.cpp with CUDA** (the system's CUDA 11.8 toolkit works with driver 535; `70` is the V100 architecture):

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/llama.cpp && cd ~/llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc -DCUDAToolkit_ROOT=/usr/local/cuda-11.8 \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 16 --target llama-server llama-bench
```

**Model file**: a GGUF at `~/models/qwen3.8-27b-Q4_K_M.gguf` (Qwen3.8 27B, Q4_K_M, 15.7 GiB).

**llama-swap**: download a release binary (`llama-swap_<ver>_linux_amd64.tar.gz`, check it against the release's `checksums.txt`) into `~/llama-swap/`, write `~/llama-swap/config.yaml`, and run it in its own tmux session:

```yaml
healthCheckTimeout: 300
logToStdout: "proxy,upstream"
startPort: 10100
models:
  "qwen3.8:27b":                      # = llm.model / chat.model in the bot configs
    env:
      - "CUDA_DEVICE_ORDER=PCI_BUS_ID"
      - "CUDA_VISIBLE_DEVICES=1,2"
      - "LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64"
      - 'LLAMA_ARG_CHAT_TEMPLATE_KWARGS={"enable_thinking":false}'
    cmd: |
      /home/USER/llama.cpp/build/bin/llama-server --host 127.0.0.1 --port ${PORT}
      -m /home/USER/models/qwen3.8-27b-Q4_K_M.gguf
      -ngl 99 -fa on -c 65536 -np 1
      --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0 --repeat-penalty 1.0 --presence-penalty 0
      --spec-type draft-mtp --spec-draft-n-max 4
      --metrics
    proxy: "http://127.0.0.1:${PORT}"
    ttl: 600
```

```bash
tmux new-session -d -s llama-swap \
  '$HOME/llama-swap/llama-swap -config $HOME/llama-swap/config.yaml -listen 127.0.0.1:8090 -watch-config 2>&1 | tee -a $HOME/llama-swap/llama-swap.log'
curl -s 127.0.0.1:8090/running        # which model is loaded
```

| Option | Why |
|---|---|
| `-c 65536 -np 1` | One slot with the whole 64K context (fits a 27b Q4 model on two 16 GB GPUs, ~10.5 and ~11.7 GB). Must match `llm.num_ctx` (paperbot) and `chat.ctx_tokens` (zoterobot); both read the real value from llama-server's `/props` after the first call. Concurrent requests queue |
| `CUDA_VISIBLE_DEVICES=1,2` + `CUDA_DEVICE_ORDER=PCI_BUS_ID` | Pins GPUs 1 and 2 with the same numbering as `nvidia-smi`; GPU 0 stays free |
| `enable_thinking: false` | Thinking is off by default on the server, and every bot also sends `chat_template_kwargs: {"enable_thinking": false}`; otherwise the model spends thousands of tokens before answering |
| sampling flags | The model's recommended defaults (from the GGUF metadata); the bots set `temperature` per request |
| `--spec-type draft-mtp` | Speculative decoding with the model's built-in MTP layer: generation 34 → 46–50 tok/s, prefill about 18% slower |
| `--metrics` | Lets the bots see queued and running requests before they unload (below) |
| `ttl: 600` | The model is unloaded 10 minutes after the last request (the zoterobot DM keep-alive). paperbot and articlebot unload right after their run |

Unloading through llama-swap (`POST /api/models/unload/<model>`) cuts off a request that is still streaming. `paperbot/llmserver.py` (used by articlebot too) therefore unloads only after llama-server reports no running or queued requests twice in a row, waiting up to 10 minutes (for example while zoterobot answers a DM), and leaves it to `ttl` otherwise. zoterobot retries a round once if its stream ends without a finish marker.

### 3.3 Tokens and keys

Secrets never go into config files. Put them in `~/.paperbot_env` (`chmod 600`). The runner scripts source it.

```bash
export S2_API_KEY="..."                 # Semantic Scholar (free; without it requests are spaced 8 s apart)
export SLACK_BOT_TOKEN="xoxb-..."       # paperbot Slack app (articlebot posts with it too)
export ZOTEROBOT_BOT_TOKEN="xoxb-..."   # zoterobot Slack app — bot token
export ZOTEROBOT_APP_TOKEN="xapp-..."   # zoterobot Slack app — app-level token (Socket Mode)
export ZOTERO_API_KEY="..."             # zotero.org/settings/keys, with write access
export ZOTERO_USER_ID="1234567"         # numeric userID from the same page (looked up from the key if wrong)
```

For Zulip add the bot's credentials to the same file (paperbot, articlebot and zoterobot all use this one bot account):

```bash
export ZULIP_EMAIL="paperbot-bot@your-zulip.example.com"   # bot e-mail from its settings page
export ZULIP_API_KEY="..."                                   # bot API key
export ZULIP_SITE="https://your-zulip.example.com"           # server URL
```

Alternatively drop the downloaded `zuliprc` at the repository root (gitignored, path in `zulip.config_file`); the environment variables win when both exist.

### 3.4 Slack apps

**paperbot app** — Bot Token Scopes: `chat:write`, `files:write` (for figures). Install, then `/invite @paperbot` in every target channel (articlebot's channel too, e.g. `#article`).

**zoterobot app** — a second app:
- *Socket Mode*: on. Create an app-level token with `connections:write` (`xapp-…`).
- *Event Subscriptions → bot events*: `app_mention`, `message.im`. Do not subscribe to events "on behalf of users".
- *Bot Token Scopes*: `chat:write`, `app_mentions:read`, `channels:history`, `groups:history`, `im:history`, `reactions:write`.
- *App Home*: enable the Messages tab and "Allow users to send messages from the messages tab".
- Reinstall, then `/invite @zoterobot` in the paper channels (it needs to read the parent message of a thread).

Both tokens of zoterobot must come from the **same** app. If they are mixed, DMs arrive but replies fail with `channel_not_found`. `python -m zoterobot --check` reports missing scopes.

### 3.5 Zulip bot (optional)

One bot account serves paperbot, articlebot and zoterobot.

- Personal settings → *Bots* → add a **Generic bot** (not incoming/outgoing webhook). Copy its e-mail and API key into `~/.paperbot_env` (`ZULIP_EMAIL`, `ZULIP_API_KEY`, `ZULIP_SITE`), or download its `zuliprc` to `~/paperbot/zuliprc` (`chmod 600`).
- Create the stream named in `zulip.stream` and subscribe the bot to it (and to any per-topic `stream` override). Streams are addressed by name; a leading `#` is ignored. Missing streams stop the run at start-up with a message naming them. Zulip topics inside the stream need no setup: a fixed `zulip_topic` is created by the first message if it does not exist, and per-paper topics are always new.
- The bot must be a **Generic bot**. An *Incoming webhook* bot can only send and fails at start-up with "This API is not available to incoming webhook bots"; the error message says so.
- paperbot: set `platform: "zulip"` on the topics that should go to Zulip (section 4.1). articlebot: `publisher.backend: zulip` (section 6.1). zoterobot: set `zulip.enabled: true` and put your Zulip e-mail (or user id) in `zulip.allowed_users` (section 5.2).
- Streaming answers edit the bot's own message. Zulip only allows edits for a realm-configured time after posting (default 10 minutes); if an edit is refused the bot posts the answer as a new message instead. Each edit also leaves an "EDITED" marker, which is why `zulip.update_every_s` defaults to 5 s.
- `python -m zoterobot --check` verifies the credentials and prints the bot's identity.

---

## 4. paperbot

### 4.1 Config (`config.yaml`)

Everything you tune is in config; see `config.example.yaml` for comments on every key.

| Section | Keys | Notes |
|---|---|---|
| `topics[]` | `name`, `platform`, `channel`, `zulip_topic`, `stream`, `interest`, `sources.semantic_scholar{query, venues, year, min_citations, sort}`, `sources.arxiv{enabled, query}`, `quota{s2, arxiv}` | `platform: slack` (default) or `zulip`. `channel` is the Slack channel. Zulip topics all post into the one stream `zulip.stream`: with `zulip_topic` every paper of the topic goes into that fixed Zulip topic (first message, full summary, figures, one paper after another); without it each paper opens its own Zulip topic named after the title. `stream` overrides the stream for one topic. Streams are checked at start-up, before any fetching. `interest` is a short English paragraph; the filter scores papers against it. `sort: citations` makes the bot page through top-cited papers it has not seen yet |
| `fetch` | `max_results_per_query`, `arxiv_lookback_days`, `s2_lookback_days`, `s2_max_pages_per_run`, `unpaywall_email` | `s2_lookback_days: 0` is recommended — let `year` bound the range |
| `quota` | `s2_per_topic`, `arxiv_per_topic`, `pending_max_age_days` | Daily cap per topic and source. Extra papers wait in `pending` and go out first the next day |
| `filter` | `embedding_threshold`, `llm_judge`, `judge_threshold` | Embedding runs on CPU in milliseconds; the judge takes ~5 s per paper on the GPU |
| `llm` | `base_url`, `model`, `num_ctx`, `max_input_chars`, `max_output_tokens`, `temperature`, `language` | `base_url` is llama-swap's `/v1`; `model` is the llama-swap model id; `num_ctx` must equal llama-server's `-c` |
| `slack` | `channel`, `attach_figure`, `max_figures`, `max_tables`, `dry_run` | Used by `platform: slack` topics. `dry_run: true` prints instead of posting |
| `zulip` | `config_file`, `stream`, `attach_figure`, `max_figures`, `max_tables`, `dry_run`, `replay`, `replay_per_run` | Used by `platform: zulip` topics. `stream` is the shared stream; `config_file` the fallback zuliprc when the `ZULIP_*` variables are not set. `replay: true` also posts the topic's papers already in the DB (see 4.4) |
| `storage` | `db_path`, `pdf_dir`, `keep_pdfs`, `chat_text_db` | Use absolute paths for a NAS. `chat_text_db` is zoterobot's text cache (`chat.text_db` in its config): when set, every published paper's full text and summary are written there so the DM chat can search and read it (see 4.4 and 5.4); empty = off |

### 4.2 Run

```bash
python -m paperbot                                              # full daily run
python -m paperbot --dry-run --topic graph-transformer --limit 1   # one topic, one paper, print only
python -m paperbot --backfill --dry-run                         # first run: ignore the arXiv lookback window
python -m paperbot --skip-filter --dry-run --limit 2            # no relevance filter (for tuning thresholds)
python -m paperbot --backfill-chat [--limit N] [--topic NAME]   # fill the chat cache with the DB's done papers (PDF re-download, no GPU, no posting)
```

`--dry-run` still records papers as `done` in the DB. To test without touching the real DB, copy it and point a temporary config at the copy with `--config`.

### 4.3 Scheduling

`scripts/loop.sh HH:MM` sleeps until the given time, checks that both GPUs have at least 13 GB free (waits up to 2 hours, then skips the day), restarts llama-swap if it is not responding, runs paperbot, and then articlebot (`scripts/articlebot.sh`, 3-hour limit, only when `articlebot/config.yaml` exists). A day skipped for lack of VRAM skips both. If the summary model is already loaded (zoterobot answered a DM a moment ago) the VRAM check is skipped and the loaded model is reused.

```bash
tmux new -d -s paperbot '~/paperbot/scripts/loop.sh 07:00'
tmux attach -t paperbot          # watch; Ctrl+b d to detach
```

Add these to `crontab -e` so everything comes back after a reboot:

```cron
@reboot tmux new-session -d -s llama-swap '$HOME/llama-swap/llama-swap -config $HOME/llama-swap/config.yaml -listen 127.0.0.1:8090 -watch-config 2>&1 | tee -a $HOME/llama-swap/llama-swap.log'
@reboot tmux new-session -d -s paperbot '$HOME/paperbot/scripts/loop.sh 07:00'
@reboot tmux new-session -d -s zoterobot '$HOME/paperbot/scripts/zoterobot.sh'
```

Config and code changes need no restart — every day is a fresh process. Changing the time, or editing `loop.sh` itself, means recreating the tmux session (bash reads a running script lazily, so replace the file rather than editing it in place).

### 4.4 Details worth knowing

- **Posting layout**: on Slack a paper is a channel message (title, authors, venue, year, citations, link, one-line summary) with the full summary and up to N figures/tables in the thread. On Zulip everything goes into one stream (`zulip.stream`) and each paper is one message — header, the summary in a collapsed spoiler, and figure/table links — posted into the topic's fixed Zulip topic (`zulip_topic`) or, without one, into a new Zulip topic named after the paper (60-character limit). A summary over 10,000 characters continues in follow-up spoiler messages.
- **Math in Zulip**: for `platform: zulip` topics the summary prompt asks for LaTeX (`$$...$$` inline, ```` ```math ```` blocks), which Zulip renders; Slack topics keep plain text. The zoterobot chat does the same for Zulip DMs.
- **Replaying DB history into Zulip** (`zulip.replay: true`, temporary — all of it lives in `paperbot/replay.py`; the pipeline only has a few lines marked `[REPLAY]`. When every old paper is in Zulip, delete `replay.py`, the lines `grep -rn "\[REPLAY\]"` finds (the two `_select` lines marked "replay_items 인자만 지운다" keep the line and lose only that argument), and the `zulip.replay*` keys, and the run is back to the original logic). For every `platform: zulip` topic the run first takes the topic's `done` papers from the DB (oldest first) and posts the ones that are not in Zulip yet, before the day's new papers. Nothing is recorded in the DB: with a fixed `zulip_topic` the bot reads that topic's messages once and looks for each paper's link (`arxiv.org/abs/…`, `doi.org/…`); without one it uses the stream's topic list. So an interrupted run simply continues next time, and once every paper is there only the normal logic remains. Each replayed paper is rebuilt from arXiv or Semantic Scholar (`sources/lookup.py`, shared with `--backfill-chat`), downloaded and summarized again (1–2 min of GPU each). `replay_per_run` (default 3) is how many old papers a topic posts per run; while a topic is replaying, its new papers are not posted but kept in `pending` (raise `quota.pending_max_age_days` if the backlog will take longer than that), and once the backlog is empty the normal quota resumes (0 = all remaining at once). A paper that cannot be rebuilt or summarized gets a short "되올리기 실패" note with its link in its topic so it is not retried; a lookup that fails because the API is down is retried next run. Logged as `Zulip 되올리기 <topic>: DB done N편 중 스트림에 없는 M편 → …`.

- **Chat cache hook** (`chatdb.py`, 2026-10-23): with `storage.chat_text_db` set, `process_paper` and the replay write each published paper into zoterobot's text cache right after posting — the full text before the `max_input_chars` cut (references stripped), the paper's topic as its "collection", and the posted summary into the cache's `summaries` table — so the DM chat can find and read papers that were never saved to Zotero/Drive. A cache failure is only a warning (the paper is already posted). `--backfill-chat` goes through the DB's `done` papers newest first (skipping those already in the cache with text): metadata from Semantic Scholar (arXiv API as fallback — it was taking 30–45 s per paper on 2026-10-23), PDF through the usual fallback chain, no figures, no GPU, 2 s pause between papers; summaries are read back from the fixed Zulip topics' messages (`parse_summaries` understands the posted layout, including split `전체 요약 (i/n)` chunks), so Slack-era papers get text but no summary until the replay re-summarizes them or the chat's `summarize_paper` makes one. It does not take the daily lockfile, so it can run beside the morning job. Logged as `[백필 i/n] <uid> — <chars>자, 요약`.
- **PDF fallback chain** (`extractor.iter_candidate_urls`): the link from the search API → the arXiv mirror → Unpaywall (needs `unpaywall_email`) → an arXiv title search. If every step fails the paper is summarized from its abstract and marked `(abstract 기반 요약)`. If there is no abstract either, nothing is posted; the paper is retried once on a later run.
- **Figures and tables** (`figures.py`, DocLayout-YOLO since 2026-10-23): the first 10 pages are rendered and run through the DocLayout-YOLO layout detector (DocStructBench weights, ~41 MB from the Hugging Face Hub cache, CPU, about 0.5 s per page). Each `figure_caption` / `table_caption` box is paired with the nearest body box of its kind (sub-panels sharing one caption are merged, table footnotes are attached), the caption text gives the label (`Figure 3`, `Table 1`), and body + caption are cropped from the page at 2x. The largest `max_figures` figures and `max_tables` tables are posted in reading order, figures first. A PDF with nothing detected falls back to a render of page 1. If the model cannot be loaded the paper is posted without attachments.
- **Relevance filter**: `cosine(interest, title + abstract) < embedding_threshold` → `filtered_out`. Survivors get a 0-10 score from the LLM with a JSON schema enforced by the server, so parsing never fails. Below `judge_threshold` → `filtered_out` with the reason stored.
- **State DB** (`store.py`), table `papers`: `done`, `filtered_out`, `pending` (full Paper JSON kept in `payload`), `expired`, `failed` (retried once), `failed_permanent`. A lockfile next to the DB prevents two runs at once.
- **Logs**: `logs/paperbot.log` (rotating) and `logs/cron.log` (runner output).

---

## 5. zoterobot

### 5.1 Zotero + Google Drive layout

zoterobot assumes the [Attanger](https://github.com/MuiseDestiny/zotero-attanger) setup: PDFs are **linked files** inside a Google Drive folder, one sub-folder per collection (nested collections → nested folders, like Attanger's `{{collection}}`), and the desktop Zotero has *Settings → Advanced → Files and Folders → Linked Attachment Base Directory* pointing at that folder. The bot writes the same structure:

```
<drive_remote>/<Collection path>/<First author> et al. - <year> - <title>.pdf     (Zotero 7 file-name format)
Zotero attachment path:  attachments:<Collection path>/<file>.pdf                    (resolved by the desktop app)

e.g. @zoterobot ICML/2026  →  Zotero collection ICML > 2026, Drive <drive_remote>/ICML/2026/, path attachments:ICML/2026/…
```

Linked files open in the desktop app only, not in the web library or mobile apps.

Drive access uses [rclone](https://rclone.org) (`~/.local/bin/rclone`): `rclone config create gdrive drive scope=drive`, then log in once in the browser. Set `library.drive_remote` to `gdrive:<folder>`.

### 5.2 Config (`zoterobot/config.yaml`)

| Section | Keys | Notes |
|---|---|---|
| top level | `allowed_users` | Slack user IDs that may use the bot. Empty = everyone is rejected; the reply shows the caller's ID |
| `zulip` | `enabled`, `config_file`, `allowed_users`, `update_every_s` | Zulip DM entry point through the paperbot bot account (`../zuliprc` by default). `allowed_users` are Zulip e-mails or numeric ids. With `enabled: true` the listener runs with Slack, or without it when no Slack tokens are set |
| `library` | `drive_remote`, `link_base`, `rclone_bin`, `tag`, `filename_lang`, `unpaywall_email` | `filename_lang: ko` gives `A 등 - 2024 - …` names like a Korean Zotero UI |
| `chat` | `llm_url`, `model`, `max_output_tokens`, `max_tool_steps`, `ctx_tokens`, `output_reserve_tokens`, `safety_tokens`, `history_max_tokens`, `chars_per_token`, `summary_enabled`, `paperbot_config`, `summaries_per_question`, `history_turns`, `vram_gpus`, `vram_min_mib`, `catalog_max`, `text_db`, `chatlog_dir`, `sync_interval_min`, `drive_scan`, `drain_timeout_s` | `llm_url` is llama-swap without `/v1`. Use the same model as paperbot so only one copy sits in VRAM; how long it stays loaded after a DM is llama-swap's `ttl`. Tool text is budgeted in tokens (see *Token budget* below): `ctx_tokens` must equal llama-server's `-c`. `summarize_paper` runs paperbot's summarizer (from `paperbot_config`) on the cached full text (PDF fetched from Drive only if the cache has none), at most `summaries_per_question` new summaries per question |

### 5.3 Use

**Save a paper (in a channel thread under a paperbot post):**

```
@zoterobot GDL                                    save the thread's paper to the "GDL" collection (created if missing)
@zoterobot ICML/2026                              save into sub-collection 2026 of ICML (Drive ICML/2026/); missing levels are created
@zoterobot GDL https://arxiv.org/abs/2505.13631   save the linked paper (works outside a thread too)
@zoterobot                                        usage + list of collections
```

Collection names match case-insensitively; sub-collections are addressed by their path (`Parent/Child`), and a bare name always means a top-level collection. The bot reacts with 👀 while working and ✅ / ❌ when done, and replies with what it did. Re-sending the same command is safe: an existing item is reused and only the missing parts (collection, PDF) are added.

**Save a paper from Zulip (DM the bot — there is no thread mention on Zulip):**

```
저장 GDL https://arxiv.org/abs/2505.13631       save the linked paper ("save" works too)
저장 ICML/2026 <quoted paper message>            in the paper's Zulip topic use "Quote and reply", then type the command under the quote
저장 GDL https://…/near/12345                    or paste the paper message's link ("Copy link to message")
저장                                             usage + list of collections
```

Any other DM is a question for the chat agent, exactly as on Slack. The bot ignores its own messages (including the daily paperbot posts) and only reads DMs, so it does not need to be mentioned.

**Ask about your library (DM the bot):**

```
graph transformer를 큰 그래프로 확장하는 논문 찾아줘          search → each hit with a 3–5 line summary
SGFormer에서 attention이 linear라고 하는 근거가 뭐야?          question → read the paper, answer from its text with section/equation references
NAGphormer랑 SGFormer 비교해줘                                comparing → commonalities and differences per aspect
How Powerful are GNNs 논문 전체 요약해줘                      full six-section summary (paperbot format), stored for later questions
GeoAI 컬렉션에 뭐 있어?                                       list a collection
좀 더 구체적으로 설명해줄래?                                   follow-up: the paper is read again before answering
새 대화                                                       forget the recent conversation
```

Progress (🔎 searching → 📖 reading → ⚖️ collecting comparison material → 📝/🧠 summarizing → answer) is streamed into one message. The answer ends with the papers it used, linked. If the GPUs do not have room for the model, the bot says so immediately instead of waiting. A new full-text summary takes 2–3 minutes; reading a whole paper for a detail question about a minute.

### 5.4 How the chat agent works

- **Question types** (the system prompt makes the model classify first): *search* ("papers about X?") → `search_papers` → each hit with a 3–5 line summary grounded in the abstract/snippet/stored summary; *question* (about one paper) → `read_paper(focus)` → answer only from the returned body text, quoting numbers, tables, equations and settings with their section/table numbers, and say so when the body does not contain it; *comparing* (several papers) → `compare_papers(aspects)` → commonalities and differences per aspect. Follow-ups asking for more detail make the model read the paper again before answering; it never asks the user to check the paper themselves.
- **Tools** (`prompts.py`): `search_papers(keywords, collection?, limit?)`, `read_paper(paper_id|title, focus?)`, `compare_papers(paper_ids|titles[], aspects?)` (splits the per-question read budget across the papers), `summarize_paper(paper_id|title, force?)`, `list_papers(collection?)`. Titles are resolved to ids by `TextCache.find_by_title`. `summarize_paper` (`summary.py`) runs paperbot's summarizer on the paper's full text from the cache (references already stripped, cut at `llm.max_input_chars`); only when the cache has no text does it download the PDF from Drive (or Zotero storage) and extract it; the result is stored in the cache (`summaries` table), reused on later questions and prepended to `read_paper` output so follow-up questions about that paper have the summary at hand. The system prompt contains the full paper list (`id | title | first author year | collections`) only when the library has at most `catalog_max` papers; the running config sets `catalog_max: 0`, so the model always finds papers through `search_papers` (or by passing a title straight to `read_paper`) rather than picking from a fixed list.
- **Full-text cache** (`textcache.py`, `sync.py`): one row per Zotero item. Text comes from Zotero's full-text index when it exists, otherwise from the PDF (downloaded from Drive via rclone, or from Zotero storage). Only items whose version changed are re-extracted. With `drive_scan: true`, PDFs that exist in the Drive folder but not in Zotero are indexed too (marked `[Drive 전용]`), with the title taken from the PDF metadata or the file name.
- **Feed documents in the same cache** (2026-10-23): paperbot (`paperbot/chatdb.py`) and articlebot (`articlebot/chatdb.py`) add every paper/article they publish — full text, metadata, and the posted summary in `summaries` — as `item_type` `paperbot_paper` (marked `[paperbot 수집]`, "collection" = the paperbot topic) or `blog_article` (`[블로그 글]`, "collection" = the organisation). Ids are `P`/`A` + 7 hex chars of the uid hash (`feed_key`), so re-runs overwrite. The Zotero sync never touches them (`known_zotero` excludes them). `search_papers` and `list_papers` take `scope`: `all` (default), `library` (Zotero + Drive), `feed`; the system prompt tells the model to narrow when the user says "내 라이브러리에서" / "봇이 올린 것 중", and answers and the "참고한 논문" footer carry the origin tag. When the same title exists in both, search returns only the library copy. `list_papers` shows at most 120 lines (newest first) and points to search. Because the other bots write from their own processes, `kick_sync` calls `refresh_if_changed()` before every DM: one `COUNT/MAX(updated_at)` query, and the in-memory docs are reloaded only when it differs. `summarize_paper` works on feed documents from the cached text; a feed document without text (abstract only) returns an error telling the model to answer from the abstract/summary.
- **Sync timing**: a sync runs in a background thread when a DM arrives (at most every `sync_interval_min`) and right after a save. DMs never wait for it; they use the current cache.
- **Token budget** (`budget.py`): instead of fixed character caps, each question gets `pool = ctx_tokens − output_reserve_tokens − safety_tokens − tokens of the messages so far` (the reserve is what is kept aside for the answer; the generation cap is still `max_output_tokens`, used whenever context remains). `read_paper` returns the whole body when it fits in the pool, otherwise a focus excerpt / head+tail cut to the pool with a note; `compare_papers` splits the pool across the papers; both never go below 4,000 chars per paper. Tokens are estimated from characters (Hangul at 1.7 chars/token, everything else at `chars_per_token`); text added since the last measurement gets +10% slack, and after every round the estimate is re-calibrated from the server's measured input tokens, `usage.prompt_tokens` including the cached prefix (logged as `라운드 n: 입력 토큰 측정 … 캐시 재사용 …`), with the calibrated ratio kept in the cache (`meta.chars_per_token`) across restarts. Defaults were set from measurements on 2026-10-21: answers peaked around 2,400 tokens (hence a 3,072 reserve), calibrated ratios were 3.42–3.54 (hence 3.4 to start), and post-calibration error stayed within ±10%. When the pool drops below one floor read (4,000 chars) the next round runs without tools, and the final generation's `max_tokens` is capped to the remaining context so nothing is silently truncated. Conversation memory is trimmed to `history_max_tokens` (oldest pairs first). The server context is read from llama-server's `/props` after the first reply and overrides `ctx_tokens` if they differ.
- **Limits are enforced in code, not by the prompt**: `history_turns` Q/A pairs per DM (and `history_max_tokens`), `max_tool_steps` tool rounds, the token budget above, `max_output_tokens` answer length, llama-swap `ttl` for unloading.
- **Chat log**: `chatlog_dir/YYYY-MM-DD.jsonl` — question, answer, tool calls, papers used, timing, and status (`ok`, `busy`, `error`, `reset`).

### 5.5 Run

```bash
python -m zoterobot --check                       # Zotero, Drive, Slack (scopes) and Zulip (zuliprc) credentials
python -m zoterobot --save GDL 2505.13631         # save one paper without Slack
python -m zoterobot --sync [--force]              # build / refresh the full-text cache (first build takes a while)
python -m zoterobot --ask "질문"                   # ask the agent from the terminal, with tool-call trace
tmux new -d -s zoterobot '~/paperbot/scripts/zoterobot.sh'   # the listener; restarts after a crash
```

A restart (SIGTERM, e.g. `pkill -f "python -m zoterobot"`) refuses new requests with a "restarting" reply, finishes the questions already being answered (up to `drain_timeout_s`, default 10 min), and only then exits; if the wait runs out, the in-flight placeholder message is replaced with an "interrupted, please ask again" note. Logs go to `logs/zoterobot.log`. The listener uses about 120 MB of RAM with an 80-paper cache (the full texts and their search-normalized copies are kept in memory) and no GPU until a DM arrives.

---

## 6. articlebot

### 6.1 Config (`articlebot/config.yaml`)

| Section | Keys | Notes |
|---|---|---|
| `sources[]` | `name`, `label`, `org`, `feed`, `channel`, `include_categories`, `exclude_categories`, `judge`, `enabled` | One RSS/Atom feed each. Sources with the same `org` share one backlog and one post a day (Meta: Engineering ML feed + Newsroom AI feed). `ai.meta.com/blog` is deliberately not used — its robots.txt forbids automated collection without written permission |
| `fetch` | `interval_days`, `lookback_days`, `min_body_chars`, `min_interval`, `timeout`, `user_agent` | Ingest every `interval_days` (3); the last ingest date is kept in the DB (`meta.last_ingest_date`), so a late paperbot run does not shift the cycle. Backlog older than `lookback_days` (90) expires |
| `judge` | `interest`, `threshold`, `excerpt_chars` | Feed summaries are poor (category names, cut at 160 chars), so the judge reads the first 3,000 chars of the body. Below `threshold` (5) → `filtered_out`, never re-scored. After changing the criteria run `--ingest-only --rejudge` |
| `report` | `per_org`, `max_attempts` | Posts per organisation per day (1); on failure the next candidate is tried, up to `max_attempts` |
| `images` | `attach`, `max_images`, `max_candidates`, `hero`, `min_width`, `min_height`, `max_download_bytes`, `max_upload_bytes`, `max_side` | More than `max_images` (4) candidates → the LLM picks from the captions (diagram > result plot/table > example). `og:image` only when the body has no image (`hero: fallback`) |
| `llm` | `base_url`, `model`, `max_input_chars`, `max_output_tokens`, `temperature`, `language`, `unload_after_run`, `peer_lockfiles` | Same server and model as paperbot. The model is not unloaded while a peer's lockfile (paperbot) is held |
| `publisher` / `slack` / `zulip` | `backend`, `dry_run` / `channel` / `stream`, `topic`, `config_file` | `backend: zulip` posts every article into one fixed stream topic (e.g. `Paper_Bot_gwlee` / `articles`) |
| `storage` | `db_path`, `preview_dir`, `chat_text_db` | `chat_text_db` = zoterobot's text cache, as in paperbot |

### 6.2 Layout

- **Slack**: channel message (title / source · date · categories · score / 🔗 link / ➤ one line), thread reply 1 with the five-section summary plus 📎 related links (arXiv, GitHub, Hugging Face, PDFs linked in the post) and 🎬 videos, thread reply 2 with up to 4 images titled by their captions.
- **Zulip**: one message per post — the same header, the summary with links and videos in a collapsed spoiler, and 🖼 uploaded-image links (the same layout code as paperbot). With the Zulip backend the summary prompt asks for LaTeX math, as in paperbot.

### 6.3 Extraction

- **Body**: per-site containers (`extractor.SITE_RULES`, XPath) with share buttons, newsletter boxes and "related posts" removed; sites without a rule fall back to `article`/`main` and a paragraph-density heuristic. MathML becomes TeX, tables become `cell | cell`, and each image leaves a `[그림 N] caption` marker so the LLM sees it in context. A page body under 400 chars falls back to the feed's `content:encoded`.
- **Images**: one candidate per `<figure>`/`<picture>`/`<img>`/video poster, lazy-load and `srcset` (largest) resolved, light/dark pairs reduced to light, icons/logos/placeholders and size variants dropped. Captions come from `figcaption`, nearby caption elements, or a descriptive alt. Images under 320×160 or with an aspect ratio over 6 are dropped; WebP/AVIF become PNG, large photos JPEG, the long side is capped at 2,400 px, animated GIFs over 8 MB keep only the first frame.

### 6.4 Run

```bash
python -m articlebot --list                    # backlog and unscored new posts per organisation (no LLM, no posting)
python -m articlebot --ingest-only             # ingest + score + backlog only (ignores the interval)
python -m articlebot --force-ingest            # ingest now regardless of the interval, then post
python -m articlebot --ingest-only --rejudge   # re-score waiting/filtered posts after changing judge.interest/threshold
python -m articlebot --dry-run [--source nvidia_tech] [--limit 1]   # print + save to preview_dir; the DB is changed only in an in-memory copy
python -m articlebot --dry-run --url <post URL>                     # one post, outside feeds and backlog
python -m articlebot --backfill-chat [--limit N]                    # re-fetch bodies of done posts into the chat cache (no LLM, no posting)
scripts/articlebot.sh                          # one real run (tokens + logs/articlebot_cron.log), as the runner does
```

State DB (`storage.db_path`), table `articles`, uid = canonical post URL: `pending` (backlog, full article JSON in `payload`, so a post can be published after it leaves the feed), `done` (summary kept), `filtered_out`, `failed` → `failed_permanent` (one retry), `expired`, `seen` (`--mark-seen`). If the bot is not in the Slack channel the run stops with `not_in_channel` and the backlog stays. Logs: `logs/articlebot.log` (rotating) and `logs/articlebot_cron.log`. Timing on V100 ×2: scoring ≈ 5 s per post, summary 30–65 s, image pick ≈ 12 s; a normal day (4 organisations) ≈ 5 minutes.

---

## 7. Operations notes

- **Shared GPU server**: `loop.sh` refuses to start the daily run when a GPU has less than 13 GB free, so it never pushes another user's job out of memory. The chat bot applies the same rule per DM.
- **Semantic Scholar 429**: happens in bursts of a few seconds even with a key; the bot backs off 3 → 5 → 10 s and retries four times.
- **arXiv 429 / 503**: the search API is capacity-limited and sometimes blocks a whole run; those days only pending arXiv papers are posted. The library's built-in retry (3 s × 3) is used as is.
- **rclone and Google**: rclone's shared Google client id is scheduled to be retired during 2026. When Google announces the date, create your own client id (`rclone.org/drive/#making-your-own-client-id`) and run `rclone config reconnect gdrive:`.
- **Zotero user ID**: it is the number on the API keys page, not the login name. If `ZOTERO_USER_ID` is not numeric the bot looks it up from the key.
- **Two Slack apps on purpose**: paperbot posts summaries; zoterobot listens. Keeping them separate means the daily job never needs Socket Mode or history scopes.

---

## 8. Code map

### paperbot/ — daily paper bot

| Module | Role |
|---|---|
| `cli.py` | Command line: arguments, logging, lockfile → `pipeline.run` |
| `pipeline.py` | One run: `fetch_all` → filter → `apply_quota` → `process_paper` → unload |
| `config.py` | `config.yaml` → frozen dataclasses, with validation |
| `models.py` | `Paper` dataclass; uid rules (arXiv id > DOI > S2 id) and canonical links |
| `sources/semantic_scholar.py` | S2 bulk search, rate limiting, citation-sorted paging with a stored cursor |
| `sources/arxiv_source.py` | arXiv search (newest first, lookback window) |
| `sources/lookup.py` | One paper's metadata by uid (arXiv / S2) — for replay and `--backfill-chat` |
| `sources/base.py` | `PaperSource` interface |
| `filter.py` | Relevance filter: embedding pre-filter (CPU) + LLM judge |
| `extractor.py` | PDF download fallback chain, text extraction, reference stripping |
| `figures.py` | Figure/table crops (DocLayout-YOLO + PyMuPDF) |
| `summarizer.py` | LLM calls (OpenAI-compatible), six-section summary |
| `prompts.py` | Summary and judge prompts, math rules per platform |
| `llmserver.py` | llama-swap helpers: loaded model, server context, safe unload — **shared** |
| `publisher/` | `__init__` (per-topic routing) · `base` · `slack` · `zulip` (client, layout, math spacing — **shared**) · `dryrun` |
| `store.py` | sqlite state: done / filtered_out / pending / expired / failed, S2 cursor |
| `chatdb.py` | Published paper → chat cache; `--backfill-chat` |
| `replay.py` | Temporary: replay Slack-era papers into Zulip (`[REPLAY]` lines mark its hooks) |

### articlebot/ — research-blog bot

| Module | Role |
|---|---|
| `cli.py` | Command line: arguments, logging, lockfile → `pipeline.run` |
| `pipeline.py` | `ingest` (feeds → judge → backlog), `report` (per-org newest → extract → summarize → publish), `--list` / `--url` / `--backfill-chat` |
| `config.py` | `articlebot/config.yaml` → frozen dataclasses, with validation |
| `models.py` | `Article`, extraction and image types; URL canonicalisation (uid) |
| `feeds.py` | RSS/Atom parsing, category filter |
| `http.py` | Polite HTTP client: per-host throttle, retries, size cap |
| `extractor.py` | Body extraction per site (`SITE_RULES`) + generic fallback; image candidates and captions |
| `images.py` | Image download, size/ratio checks, format conversion, resizing |
| `llm.py` | Judge, summary and image-pick calls; unload via `paperbot.llmserver` |
| `prompts.py` | Judge, summary and image-pick prompts |
| `publisher/` | `__init__` (backend choice) · `base` · `slack` · `zulip` (uses `paperbot.publisher.zulip`) · `dryrun` |
| `store.py` | sqlite state: pending (backlog) / done / filtered_out / failed / expired / seen |
| `chatdb.py` | Published post → chat cache (via `zoterobot.textcache`) |
| `lockfile.py` | One run at a time; is paperbot running (then skip the unload) |

### zoterobot/ — Zotero + chat bot

| Module | Role |
|---|---|
| `cli.py` | Command line: run the listener, or `--check` / `--save` / `--sync` / `--ask` |
| `listener.py` | Slack Socket Mode + Zulip DM routing (`MultiBot`), worker queues, background sync |
| `handlers.py` | `SaveHandler` (save command) and `ChatHandler` (DM → agent) |
| `slackbot.py` / `zulipbot.py` | Platform layer: reply, react, update, parent message, allow-list |
| `messages.py` | Command parsing and reply formatting per platform |
| `resolver.py` | Message text → arXiv/DOI/S2 id → paper metadata |
| `library.py` | Save flow: collection → item → PDF → Drive → linked attachment |
| `zotero.py` | Zotero Web API client, item JSON, file-name rules |
| `drive.py` | rclone wrapper: upload, download, list |
| `textcache.py` | Chat cache (sqlite): documents, search, reading, stored summaries — **shared** |
| `sync.py` | Fills the cache from Zotero and Drive-only PDFs |
| `agent.py` | Tool-calling loop over llama-swap (SSE streaming) |
| `budget.py` | Token budget per question, estimate ↔ measured calibration |
| `summary.py` | `summarize_paper`: paperbot's summarizer on cached full text |
| `prompts.py` | Tool definitions and the system prompt |
| `gpu.py` | Free-VRAM check before loading the model |
| `session.py` | Per-DM conversation memory, JSONL chat log |
| `config.py` | `zoterobot/config.yaml` → dataclasses |

### scripts/

| Script | Role |
|---|---|
| `loop.sh` | Daily runner: wait for HH:MM → VRAM check → llama-swap health → paperbot → articlebot |
| `articlebot.sh` | One articlebot run (loads tokens, logs to `logs/articlebot_cron.log`) |
| `zoterobot.sh` | Keeps the zoterobot listener running (restarts it after a crash) |

---

MIT License — see `LICENSE`.
