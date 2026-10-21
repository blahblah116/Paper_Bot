# paperbot & zoterobot

Two Slack bots for a research group, running fully on a local GPU server with [Ollama](https://ollama.com) — no external LLM API.

| Bot | What it does | When it runs | GPU |
|---|---|---|---|
| **paperbot** | Finds new papers on Semantic Scholar and arXiv, filters them by your interests, summarizes each one with a local LLM, and posts to Slack channels | Once a day (tmux runner, default 07:00) | During the run only |
| **zoterobot** | Saves a posted paper to your Zotero library + Google Drive when you mention it in a thread, and answers DMs about your library (a tool-calling chat agent) | Always on (Socket Mode listener) | Only while answering a DM |

Both live in this repo and share one virtualenv and one Ollama model. They are separate Slack apps with separate tokens.

---

## Contents

1. [How it works](#1-how-it-works)
2. [Repository layout](#2-repository-layout)
3. [Setup](#3-setup)
4. [paperbot](#4-paperbot)
5. [zoterobot](#5-zoterobot)
7. [Operations notes](#7-operations-notes)

---

## 1. How it works

### paperbot — one daily run (`python run.py`)

```
scripts/loop.sh (tmux, 07:00) ── checks free VRAM, restarts Ollama if dead ──▶ run.py
  │
  ├─ 1. check Slack config      token / channel; stop early if wrong (no paper is consumed)
  ├─ 2. fetch                   topic × [Semantic Scholar → arXiv]; one source failing does not stop the others
  ├─ 3. dedupe                  uid = arXiv id > DOI > S2 id; skip papers already in the DB
  ├─ 4. relevance filter        embedding cosine → LLM judge (0-10, JSON schema enforced)
  ├─ 5. pending queue + quota   old pending papers first (FIFO), then new ones, up to the daily cap per topic/source
  ├─ 6. per paper               get PDF → extract text + figures → summarize → post to Slack → mark done
  └─ 7. finish                  unload the model (free the GPU), write a summary line to the log
```

Each paper becomes a **channel message** (title, authors, venue, year, citations, link, one-line summary) plus a **thread reply** with the full six-section summary and up to N figures.

### zoterobot — two inputs, one process

```
Slack thread: "@zoterobot GDL"          Slack DM: "which paper used Laplacian eigenvectors as PE?"
        │                                        │
  SaveHandler                              ChatHandler
  ├─ find the paper from the parent         ├─ GPU free? (else reply "busy")
  │   message's arXiv/DOI link              ├─ post "thinking…" and start a background library sync
  ├─ look up metadata (S2 → arXiv)          ├─ Agent: qwen calls tools — search_library / read_paper / list_papers
  ├─ find or create the Zotero collection   │   over the local full-text cache; progress is streamed into the message
  ├─ download the PDF → rclone → Drive      └─ final answer + "papers used" with links; everything goes to the chat log
  └─ Zotero item + linked-file attachment
```

The chat agent has **no embedding model**: the LLM decides which tool to call, with English keywords it derives from the (Korean) question. The full-text cache is filled from Zotero's full-text index or, for Drive-linked PDFs, by downloading and extracting the PDF.

---

## 2. Repository layout

```
paperbot/                    daily summary bot
├── cli.py                   command line (args, logging, lockfile) → pipeline.run
├── pipeline.py              one run: fetch_all → filter → quota (apply_quota) → process_paper
├── config.py                config.yaml → frozen dataclasses, with validation
├── models.py                Paper dataclass and uid rules (arXiv id > DOI > S2 id)
├── sources/
│   ├── base.py              PaperSource interface
│   ├── semantic_scholar.py  S2 bulk search, rate limiting, citation-sorted paging with a stored cursor
│   └── arxiv_source.py      arXiv search (newest first, lookback window)
├── filter.py                relevance filter: embedding pre-filter (CPU) + LLM judge
├── extractor.py             PDF download fallback chain, text extraction, reference stripping
├── figures.py               figure extraction (caption-anchored) for Slack attachments
├── summarizer.py            Ollama chat calls, six-section summary, model unload
├── prompts.py               summary and judge prompts
├── publisher.py             Slack posting (channel + thread + figures) and the dry-run printer
└── store.py                 sqlite state: done / filtered_out / pending / expired / failed, S2 cursor

zoterobot/                   Zotero + chat bot (always on)
├── cli.py                   command line: run the listener, or --check / --save / --sync / --ask
├── listener.py              Socket Mode routing, worker queues, background sync
├── handlers.py              SaveHandler (thread mention → save), ChatHandler (DM → agent)
├── slackbot.py              small Slack Web API helper (reply, react, update, parent message, allow-list)
├── messages.py              command parsing and reply formatting (pure functions)
├── resolver.py              Slack text → arXiv/DOI/S2 id → paper metadata (S2 → arXiv → title fallback)
├── zotero.py                Zotero Web API client, item JSON and file-name rules
├── drive.py                 rclone wrapper (upload, download, list, check)
├── library.py               the save flow: collection → item → PDF → Drive → linked attachment
├── textcache.py             full-text cache (sqlite): documents, keyword search, reading
├── sync.py                  fills the cache from Zotero (and PDFs that exist only in the Drive folder)
├── agent.py                 tool-calling loop over Ollama /api/chat
├── prompts.py               tool definitions and the system prompt
├── gpu.py                   free-VRAM check before loading the model
├── session.py               per-DM conversation memory and the JSONL chat log
└── config.py                zoterobot/config.yaml → dataclasses

run.py                       paperbot entry point (same as `python -m paperbot`)
scripts/loop.sh              tmux runner: sleep until HH:MM, VRAM check, run paperbot, run after_run.sh
scripts/zoterobot.sh         tmux runner for the listener (restarts it if it dies)
scripts/after_run.example.sh optional hook that runs right after paperbot (copy to after_run.sh)
config.example.yaml          paperbot settings template
zoterobot/config.example.yaml zoterobot settings template
```

---

## 3. Setup

### 3.1 Python

Python 3.10+. Install the CPU build of torch first so the embedding model does not pull the CUDA wheels.

```bash
cd ~/paperbot
python3 -m venv venv && source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cp config.example.yaml config.yaml
cp zoterobot/config.example.yaml zoterobot/config.yaml
```

### 3.2 Ollama and the model

Ollama runs as a normal user from `~/ollama` (no sudo). Keep it in its own tmux session.

```bash
curl -fL https://ollama.com/download/ollama-linux-amd64.tar.zst -o /tmp/ollama.tar.zst
mkdir -p ~/ollama && tar --zstd -xf /tmp/ollama.tar.zst -C ~/ollama
tmux new-session -d -s ollama \
  'CUDA_VISIBLE_DEVICES=1,2 GGML_VK_VISIBLE_DEVICES=1,2 OLLAMA_CONTEXT_LENGTH=65536 \
   $HOME/ollama/bin/ollama serve 2>&1 | tee -a $HOME/ollama/serve.log'
~/ollama/bin/ollama pull qwen3.8:27b
```

| Variable | Why |
|---|---|
| `OLLAMA_CONTEXT_LENGTH=65536` | Without it Ollama sizes the context from total VRAM and spreads the KV cache over every GPU. 64K keeps a 27b model on two 16GB GPUs (~21GB). It must match `llm.num_ctx` in config |
| `GGML_VK_VISIBLE_DEVICES=1,2` | Pins the GPUs for the Vulkan backend (used when the NVIDIA driver is too old for the CUDA backend) |
| `CUDA_VISIBLE_DEVICES=1,2` | Same pin for the CUDA backend |

Thinking is always turned off in code (`reasoning_effort="none"` / `think=false`); otherwise the model spends thousands of tokens before answering.

### 3.3 Tokens and keys

Secrets never go into config files. Put them in `~/.paperbot_env` (`chmod 600`). Both runner scripts source it.

```bash
export S2_API_KEY="..."                 # Semantic Scholar (free; without it requests are spaced 8 s apart)
export SLACK_BOT_TOKEN="xoxb-..."       # paperbot Slack app
export ZOTEROBOT_BOT_TOKEN="xoxb-..."   # zoterobot Slack app — bot token
export ZOTEROBOT_APP_TOKEN="xapp-..."   # zoterobot Slack app — app-level token (Socket Mode)
export ZOTERO_API_KEY="..."             # zotero.org/settings/keys, with write access
export ZOTERO_USER_ID="1234567"         # numeric userID from the same page (looked up from the key if wrong)
```

### 3.4 Slack apps

**paperbot app** — Bot Token Scopes: `chat:write`, `files:write` (for figures). Install, then `/invite @paperbot` in every target channel.

**zoterobot app** — a second app:
- *Socket Mode*: on. Create an app-level token with `connections:write` (`xapp-…`).
- *Event Subscriptions → bot events*: `app_mention`, `message.im`. Do not subscribe to events "on behalf of users".
- *Bot Token Scopes*: `chat:write`, `app_mentions:read`, `channels:history`, `groups:history`, `im:history`, `reactions:write`.
- *App Home*: enable the Messages tab and "Allow users to send messages from the messages tab".
- Reinstall, then `/invite @zoterobot` in the paper channels (it needs to read the parent message of a thread).

Both tokens of zoterobot must come from the **same** app. If they are mixed, DMs arrive but replies fail with `channel_not_found`. `python -m zoterobot --check` reports missing scopes.

---

## 4. paperbot

### 4.1 Config (`config.yaml`)

Everything you tune is in config; see `config.example.yaml` for comments on every key.

| Section | Keys | Notes |
|---|---|---|
| `topics[]` | `name`, `channel`, `interest`, `sources.semantic_scholar{query, venues, year, min_citations, sort}`, `sources.arxiv{enabled, query}`, `quota{s2, arxiv}` | `interest` is a short English paragraph; the filter scores papers against it. `sort: citations` makes the bot page through top-cited papers it has not seen yet |
| `fetch` | `max_results_per_query`, `arxiv_lookback_days`, `s2_lookback_days`, `s2_max_pages_per_run`, `unpaywall_email` | `s2_lookback_days: 0` is recommended — let `year` bound the range |
| `quota` | `s2_per_topic`, `arxiv_per_topic`, `pending_max_age_days` | Daily cap per topic and source. Extra papers wait in `pending` and go out first the next day |
| `filter` | `embedding_threshold`, `llm_judge`, `judge_threshold` | Embedding runs on CPU in milliseconds; the judge takes ~5 s per paper on the GPU |
| `llm` | `model`, `num_ctx`, `max_input_chars`, `max_output_tokens`, `temperature`, `language` | `num_ctx` must equal the server's `OLLAMA_CONTEXT_LENGTH` |
| `slack` | `channel`, `attach_figure`, `max_figures`, `dry_run` | `dry_run: true` prints instead of posting |
| `storage` | `db_path`, `pdf_dir`, `keep_pdfs` | Use absolute paths for a NAS |

### 4.2 Run

```bash
python run.py                                              # full daily run
python run.py --dry-run --topic graph-transformer --limit 1   # one topic, one paper, print only
python run.py --backfill --dry-run                         # first run: ignore the arXiv lookback window
python run.py --skip-filter --dry-run --limit 2            # no relevance filter (for tuning thresholds)
```

`--dry-run` still records papers as `done` in the DB. To test without touching the real DB, copy it and point a temporary config at the copy with `--config`.

### 4.3 Scheduling

`scripts/loop.sh HH:MM` sleeps until the given time, checks that both GPUs have at least 13 GB free (waits up to 2 hours, then skips the day), restarts Ollama if it is not responding, runs `run.py`, and finally runs `scripts/after_run.sh` if it exists. If the summary model is already loaded (zoterobot answered a DM a moment ago) the VRAM check is skipped and the loaded model is reused.

```bash
tmux new -d -s paperbot '~/paperbot/scripts/loop.sh 07:00'
tmux attach -t paperbot          # watch; Ctrl+b d to detach
```

Add these to `crontab -e` so everything comes back after a reboot:

```cron
@reboot tmux new-session -d -s ollama 'CUDA_VISIBLE_DEVICES=1,2 GGML_VK_VISIBLE_DEVICES=1,2 OLLAMA_CONTEXT_LENGTH=65536 $HOME/ollama/bin/ollama serve 2>&1 | tee -a $HOME/ollama/serve.log'
@reboot tmux new-session -d -s paperbot '$HOME/paperbot/scripts/loop.sh 07:00'
@reboot tmux new-session -d -s zoterobot '$HOME/paperbot/scripts/zoterobot.sh'
```

Config and code changes need no restart — every day is a fresh process. Changing the time means recreating the tmux session and editing the crontab line.

### 4.4 Details worth knowing

- **PDF fallback chain** (`extractor.iter_candidate_urls`): the link from the search API → the arXiv mirror → Unpaywall (needs `unpaywall_email`) → an arXiv title search. If every step fails the paper is summarized from its abstract and marked `(abstract 기반 요약)`. If there is no abstract either, nothing is posted; the paper is retried once on a later run.
- **Figures** (`figures.py`): the first 8 pages are scanned for `Figure N` captions; the drawings, images and short labels above a caption in the same column are rendered together at 2x. Tables and page headers are excluded.
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
| `library` | `drive_remote`, `link_base`, `rclone_bin`, `tag`, `filename_lang`, `unpaywall_email` | `filename_lang: ko` gives `A 등 - 2024 - …` names like a Korean Zotero UI |
| `chat` | `model`, `keep_alive`, `max_output_tokens`, `max_tool_steps`, `ctx_tokens`, `output_reserve_tokens`, `safety_tokens`, `history_max_tokens`, `chars_per_token`, `summary_enabled`, `paperbot_config`, `summaries_per_question`, `history_turns`, `vram_gpus`, `vram_min_mib`, `catalog_max`, `text_db`, `chatlog_dir`, `sync_interval_min`, `drive_scan` | Use the same model as paperbot so only one copy sits in VRAM. Tool text is budgeted in tokens (see *Token budget* below): `ctx_tokens` must equal the server's `OLLAMA_CONTEXT_LENGTH`. `summarize_paper` runs paperbot's summarizer (from `paperbot_config`) on the cached full text (PDF fetched from Drive only if the cache has none), at most `summaries_per_question` new summaries per question |

### 5.3 Use

**Save a paper (in a channel thread under a paperbot post):**

```
@zoterobot GDL                                    save the thread's paper to the "GDL" collection (created if missing)
@zoterobot ICML/2026                              save into sub-collection 2026 of ICML (Drive ICML/2026/); missing levels are created
@zoterobot GDL https://arxiv.org/abs/2505.13631   save the linked paper (works outside a thread too)
@zoterobot                                        usage + list of collections
```

Collection names match case-insensitively; sub-collections are addressed by their path (`Parent/Child`), and a bare name always means a top-level collection. The bot reacts with 👀 while working and ✅ / ❌ when done, and replies with what it did. Re-sending the same command is safe: an existing item is reused and only the missing parts (collection, PDF) are added.

**Ask about your library (DM the bot):**

```
Which paper used Laplacian eigenvectors as positional encoding?
What is in the GeoAI collection?
How does NAGphormer turn a node into tokens?
새 대화                                            forget the recent conversation
```

Progress (🔎 searching → 📖 reading → answer) is streamed into one message. The answer ends with the papers it used, linked. If the GPUs do not have room for the model, the bot says so immediately instead of waiting.

### 5.4 How the chat agent works

- **Question types** (the system prompt makes the model classify first): *search* ("papers about X?") → `search_papers` → each hit with a 3–5 line summary grounded in the abstract/snippet/stored summary; *question* (about one paper) → `read_paper(focus)` → answer only from the returned body text, say so when the body does not contain it; *comparing* (several papers) → `compare_papers(aspects)` → commonalities and differences per aspect.
- **Tools** (`prompts.py`): `search_papers(keywords, collection?, limit?)`, `read_paper(paper_id|title, focus?)`, `compare_papers(paper_ids|titles[], aspects?)` (splits the per-question read budget across the papers), `summarize_paper(paper_id|title, force?)`, `list_papers(collection?)`. Titles are resolved to ids by `TextCache.find_by_title`. `summarize_paper` (`summary.py`) runs paperbot's summarizer on the paper's full text from the cache (references already stripped, cut at `llm.max_input_chars`); only when the cache has no text does it download the PDF from Drive (or Zotero storage) and extract it; the result is stored in the cache (`summaries` table), reused on later questions and prepended to `read_paper` output so follow-up questions about that paper have the summary at hand. The system prompt contains the full paper list (`id | title | first author year | collections`) only when the library has at most `catalog_max` papers; the running config sets `catalog_max: 0`, so the model always finds papers through `search_papers` (or by passing a title straight to `read_paper`) rather than picking from a fixed list.
- **Full-text cache** (`textcache.py`, `sync.py`): one row per Zotero item. Text comes from Zotero's full-text index when it exists, otherwise from the PDF (downloaded from Drive via rclone, or from Zotero storage). Only items whose version changed are re-extracted. With `drive_scan: true`, PDFs that exist in the Drive folder but not in Zotero are indexed too (marked `[Drive 전용]`), with the title taken from the PDF metadata or the file name.
- **Sync timing**: a sync runs in a background thread when a DM arrives (at most every `sync_interval_min`) and right after a save. DMs never wait for it; they use the current cache.
- **Token budget** (`budget.py`): instead of fixed character caps, each question gets `pool = ctx_tokens − output_reserve_tokens − safety_tokens − tokens of the messages so far` (the reserve is what is kept aside for the answer; the generation cap is still `max_output_tokens`, used whenever context remains). `read_paper` returns the whole body when it fits in the pool, otherwise a focus excerpt / head+tail cut to the pool with a note; `compare_papers` splits the pool across the papers; both never go below 4,000 chars per paper. Tokens are estimated from characters (Hangul at 1.7 chars/token, everything else at `chars_per_token`); text added since the last measurement gets +10% slack, and after every round the estimate is re-calibrated from Ollama's real `prompt_eval_count` (logged as `라운드 n: 입력 토큰 측정 …`), with the calibrated ratio kept in the cache (`meta.chars_per_token`) across restarts. Defaults were set from measurements on 2026-10-21: answers peaked around 2,400 tokens (hence a 3,072 reserve), calibrated ratios were 3.42–3.54 (hence 3.4 to start), and post-calibration error stayed within ±10%. When the pool drops below one floor read (4,000 chars) the next round runs without tools, and the final generation's `num_predict` is capped to the remaining context so nothing is silently truncated. Conversation memory is trimmed to `history_max_tokens` (oldest pairs first). The server context is read from `/api/ps` after the first reply and overrides `ctx_tokens` if they differ.
- **Limits are enforced in code, not by the prompt**: `history_turns` Q/A pairs per DM (and `history_max_tokens`), `max_tool_steps` tool rounds, the token budget above, `max_output_tokens` answer length, Ollama `keep_alive` for unloading.
- **Chat log**: `chatlog_dir/YYYY-MM-DD.jsonl` — question, answer, tool calls, papers used, timing, and status (`ok`, `busy`, `error`, `reset`).

### 5.5 Run

```bash
python -m zoterobot --check                       # Zotero, Drive and Slack credentials + missing scopes
python -m zoterobot --save GDL 2505.13631         # save one paper without Slack
python -m zoterobot --sync [--force]              # build / refresh the full-text cache (first build takes a while)
python -m zoterobot --ask "질문"                   # ask the agent from the terminal, with tool-call trace
tmux new -d -s zoterobot '~/paperbot/scripts/zoterobot.sh'   # the listener; restarts after a crash
```

Logs go to `logs/zoterobot.log`. The listener uses about 60 MB of RAM and no GPU until a DM arrives.

---

## 7. Operations notes

- **Shared GPU server**: `loop.sh` refuses to start the daily run when a GPU has less than 13 GB free, so it never pushes another user's job out of memory. The chat bot applies the same rule per DM.
- **Semantic Scholar 429**: happens in bursts of a few seconds even with a key; the bot backs off 3 → 5 → 10 s and retries four times.
- **arXiv 429 / 503**: the search API is capacity-limited and sometimes blocks a whole run; those days only pending arXiv papers are posted. The library's built-in retry (3 s × 3) is used as is.
- **rclone and Google**: rclone's shared Google client id is scheduled to be retired during 2026. When Google announces the date, create your own client id (`rclone.org/drive/#making-your-own-client-id`) and run `rclone config reconnect gdrive:`.
- **Zotero user ID**: it is the number on the API keys page, not the login name. If `ZOTERO_USER_ID` is not numeric the bot looks it up from the key.
- **Two Slack apps on purpose**: paperbot posts summaries; zoterobot listens. Keeping them separate means the daily job never needs Socket Mode or history scopes.

---

MIT License — see `LICENSE`.
