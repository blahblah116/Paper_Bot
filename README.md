# paperbot

> Local-LLM paper summary bot — Semantic Scholar + arXiv에서 논문을 수집하고,
> relevance 필터를 거쳐 Ollama(qwen3.8:27b)로 구조화 요약을 만들어 Slack/Zulip에 포스팅한다.

외부 LLM API 없이 연구실 GPU 서버(V100 ×2, Vulkan)에서 전부 돌아간다.
주제·학회·연도·인용수·필터 강도·발행량은 모두 `config.yaml`에서 조절한다.

---

## 목차

1. [파이프라인](#1-파이프라인)
2. [설치](#2-설치)
3. [Ollama와 모델](#3-ollama와-모델)
4. [설정 (config.yaml)](#4-설정-configyaml)
5. [검색 소스](#5-검색-소스)
6. [relevance 필터](#6-relevance-필터)
7. [발행 쿼터와 pending 대기열](#7-발행-쿼터와-pending-대기열)
8. [PDF 확보·추출·요약](#8-pdf-확보추출요약)
9. [메신저 발행](#9-메신저-발행)
10. [실행과 테스트](#10-실행과-테스트)
11. [주기 실행](#11-주기-실행)
12. [데이터·로그·상태](#12-데이터로그상태)
13. [운영 메모](#13-운영-메모)

---

## 1. 파이프라인

`run.py` 한 번 실행(=하루 1회)에서 일어나는 일:

```
loop.sh (tmux, 매일 07:00) ── Ollama 생존 확인 ──▶ run.py
  │
  ├─ ① 발행 설정 검증        Slack 토큰·채널 확인. 실패 시 즉시 종료 (논문 소모 없음)
  ├─ ② 검색                  topic × [Semantic Scholar → arXiv], 소스 단위 실패 격리
  ├─ ③ dedupe + 미처리 추출   uid(arXiv id > DOI > S2 id)로 통합, DB에 없는 것만
  ├─ ④ relevance 필터        임베딩 cosine → LLM judge(0~10, JSON 스키마 강제)
  ├─ ⑤ pending 대기열        만료 폐기 → topic×소스별 FIFO 로드
  ├─ ⑥ 쿼터                  [대기열] + [신규] 순으로 상한까지 선택, 남은 신규 → pending
  ├─ ⑦ 논문별 처리           PDF 확보 → 텍스트·figure 추출 → 요약 → 발행 → done
  └─ ⑧ 마무리                모델 언로드(GPU 반납), 집계 로그
```

| 단계 | 모듈 | 핵심 동작 |
|---|---|---|
| ② 검색 | `sources/semantic_scholar.py`, `sources/arxiv_source.py` | 소스별 최신순 `max_results_per_query`편. S2 429는 3·5·10초 백오프 4회, arXiv는 라이브러리 재시도(3초×3) |
| ③ dedupe | `models.py`, `store.py` | `make_uid()`가 버전·arXiv 자동 DOI를 정규화. `done/filtered_out/pending/expired/failed_permanent`는 제외, `failed`는 재시도 |
| ④ 필터 | `filter.py` | 임베딩 threshold 미만 → `filtered_out`. judge는 thinking 끄고 `{"score","reason"}` 스키마로 채점, threshold 미만 → `filtered_out` |
| ⑤⑥ 쿼터 | `run.py apply_quota()`, `store.py` | 대기열이 먼저 소진되고 신규 최신 논문이 남은 자리를 채움. 초과 신규분은 `pending`(Paper 전체 JSON 보존) |
| ⑦ 처리 | `extractor.py`, `summarizer.py`, `publisher.py` | 폴백 체인으로 PDF 확보, References 제거, 캡션 기준 figure 선별, 6섹션 한국어 요약, 채널+스레드 발행 |

---

## 2. 설치

요구 사항: Python 3.10+, 상시 실행 Ollama(`http://localhost:11434/v1`),
Semantic Scholar API 키(권장), Slack 봇 토큰 또는 Zulip `zuliprc`(실전송 시).

```bash
cd ~/paperbot
python3 -m venv venv && source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # 임베딩은 CPU, CUDA 휠(~3GB) 회피
pip install -r requirements.txt
cp config.example.yaml config.yaml                                   # 실제 config.yaml은 gitignore
```

토큰·키는 config에 넣지 않고 `~/.paperbot_env`(권한 600)에 둔다. `loop.sh`가 실행 직전에 읽는다.

```bash
# ~/.paperbot_env
export SLACK_BOT_TOKEN="xoxb-..."
export S2_API_KEY="..."
```

---

## 3. Ollama와 모델

### 설치·실행 (sudo 없이 유저 홈)

```bash
curl -fL https://ollama.com/download/ollama-linux-amd64.tar.zst -o /tmp/ollama.tar.zst
mkdir -p ~/ollama && tar --zstd -xf /tmp/ollama.tar.zst -C ~/ollama
echo 'export PATH="$HOME/ollama/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc

tmux new-session -d -s ollama \
  'CUDA_VISIBLE_DEVICES=1,2 GGML_VK_VISIBLE_DEVICES=1,2 OLLAMA_CONTEXT_LENGTH=65536 \
   $HOME/ollama/bin/ollama serve 2>&1 | tee -a $HOME/ollama/serve.log'
```

| 환경변수 | 이유 |
|---|---|
| `OLLAMA_CONTEXT_LENGTH=65536` | 미설정 시 전체 VRAM 기준으로 컨텍스트가 자동 확장(262144)되어 KV 캐시가 GPU 3장에 27GB로 퍼진다. 64K면 27b 기준 GPU 2장에 약 21GB |
| `GGML_VK_VISIBLE_DEVICES=1,2` | Vulkan 백엔드의 GPU 고정. Vulkan은 `CUDA_VISIBLE_DEVICES`를 무시한다 |
| `CUDA_VISIBLE_DEVICES=1,2` | 드라이버 업데이트로 CUDA 백엔드가 켜질 경우 대비 |

> **num_ctx**: 봇은 매 요청에 `num_ctx`를 넘기지만 Ollama 0.34의 OpenAI 호환 엔드포인트는 이를 무시한다.
> 실효값은 서버의 `OLLAMA_CONTEXT_LENGTH`이며, 봇은 첫 호출 후 `/api/ps`로 실제 컨텍스트를 확인해
> config보다 작으면 경고를 남긴다.

### 모델

```bash
ollama pull qwen3.8:27b      # Q4_K_M 16.5GB — 현재 운영 모델
```

| 항목 | 값 (V100 ×2, Vulkan 실측) |
|---|---|
| 배치 | GPU 1·2에 66/66 레이어, 가중치 15GB + 64K KV 4GB |
| 요약 1편 (전문 40~46k자) | 약 100~150초 |
| judge 1편 | 약 5초 |

- **thinking은 코드에서 항상 꺼진다** (`summarizer.chat()`이 `reasoning_effort="none"` 전송).
  Qwen3.8의 기본 추론 강도(xhigh)는 단순 요약에도 수천 토큰을 생각에 쓰고, 그 토큰이
  `max_output_tokens`를 소진하면 빈 응답이 나온다. Ollama 0.34 실측상 `think` 옵션은 무시되고
  `reasoning_effort`만 동작한다.
- 8비트(q8_0/mxfp8)는 가중치만 28GB라 GPU 2장(32GB)에 들어가지 않는다. V100은 FP8 연산 유닛도 없다.

### V100 메모

bf16·FlashAttention-2·AWQ 미지원 → GGUF(Ollama)만 사용, vLLM은 쓰지 않는다.
드라이버 535가 CUDA 백엔드 요구(550+)보다 낮아 **Vulkan으로 동작**한다. 임베딩 모델은 CPU.

---

## 4. 설정 (config.yaml)

### topics

```yaml
topics:
  - name: "graph-transformer"
    channel: "#graph_transformer"        # 이 topic의 발행 대상 (생략 시 slack.channel / zulip.stream)
    interest: >                          # relevance 필터의 기준이 되는 자연어 서술
      Graph Transformer and Graph Mamba–family architectures: ...
    sources:
      semantic_scholar:
        query: '"graph transformer" | "graph mamba"'    # S2 bulk 검색 문법
        venues: ["ICLR", "NeurIPS", "ICML", "KDD"]      # 빈 리스트 = venue 필터 없음
        year: "2020-"                                   # "2024" | "2024-2026" | "2024-"
        min_citations: 5
        sort: "recency"                                 # recency(기본) | citations(백필용)
      arxiv:
        enabled: true
        query: '(cat:cs.AI OR cat:cs.LG) AND (abs:"graph transformer" OR abs:"graph mamba")'
```

### 나머지 섹션

| 섹션 | 키 | 현재 값 | 설명 |
|---|---|---|---|
| `fetch` | `max_results_per_query` | 10 | 소스별 서버 요청 크기(최신순). DB 제외는 그 뒤 → 신규 후보 = 10 − 이미 아는 논문 |
| | `arxiv_lookback_days` | 14 | arXiv 제출일 창. arXiv엔 서버측 연도 필터가 없어 필수(≥1) |
| | `s2_lookback_days` | 0 | 0 = 비활성, 시간 범위를 topic `year`에 위임 (권장) |
| | `unpaywall_email` | 설정됨 | PDF 403 시 Unpaywall OA 사본 조회. 비우면 생략 |
| `quota` | `s2_per_topic` / `arxiv_per_topic` | 2 / 1 | 실행당 채널·소스별 발행 상한. 0 = 무제한 |
| | `pending_max_age_days` | 30 | 이보다 오래 대기한 pending 폐기. 0 = 무제한 |
| `filter` | `embedding_model` / `embedding_device` | MiniLM-L6-v2 / cpu | 임베딩 프리필터 |
| | `embedding_threshold` | 0.25 | interest↔title+abstract cosine 하한 |
| | `llm_judge` / `judge_threshold` | true / 6 | LLM judge 사용 여부와 0~10 점수 하한 |
| `llm` | `model` | qwen3.8:27b | Ollama 모델 태그 |
| | `num_ctx` / `max_input_chars` | 65536 / 150000 | 컨텍스트, 본문 입력 상한(≈40k 토큰, 초과 시 앞70%+뒤30%) |
| | `max_output_tokens` / `temperature` | 8192 / 0.3 | 출력 상한(thinking 꺼짐 → 순수 출력), 샘플링 온도 |
| | `language` | ko | 요약 언어 |
| `publisher` | `backend` / `dry_run` | slack / false | 발행 백엔드, dry_run=true면 stdout |
| `slack` | `channel` / `attach_figure` / `max_figures` | 기본 채널 / true / 5 | 기본 채널, figure 첨부 여부·상한 |
| `zulip` | `config_file` / `stream` | ./zuliprc / papers | Zulip 백엔드용 |
| `storage` | `db_path` / `pdf_dir` / `keep_pdfs` | NAS 경로 / NAS 경로 / false | 상태 DB, PDF 임시 저장, 요약 후 PDF 삭제 |

> `./`로 시작하는 storage 경로는 paperbot 디렉토리 기준 상대 경로다. NAS는 절대 경로로 쓴다.

---

## 5. 검색 소스

topic마다 **S2 → arXiv** 순서로 부른다. uid가 겹치면 메타데이터가 풍부한 S2 레코드가 남는다.
한 소스가 실패해도 나머지는 계속 진행한다.

### Semantic Scholar (주 소스)

- bulk 검색, venue·연도·최소 인용수 필터, `publicationDate` 내림차순. lookback에 닿으면 페이지네이션 조기 중단.
- API 키가 있으면 요청 간격 1.2초, 없으면 8초(공용 풀). 키는 https://www.semanticscholar.org/product/api 에서 무료 발급.
- **429 백오프: 3초 → 5초 → 10초, 최대 4회 시도.** S2의 429는 몇 초 단위로 열리고 닫혀서 이 간격이면 거의 잡힌다. `Retry-After`가 있으면 그 값을 따른다(상한 120초). 401/403은 키 문제라 즉시 실패.
- venue 이름은 축약형(ICLR, NeurIPS, CVPR 등)이 매칭된다. 마이너 학회는 아래로 확인:
  ```bash
  curl -s "https://api.semanticscholar.org/graph/v1/paper/search/bulk?query=QUERY&venue=VENUE&fields=title,venue" | head -40
  ```
- 신선도 특성: proceedings 인덱싱이 수개월 늦고 버스트로 들어오며, `min_citations`를 쓰면 인용 문턱을 넘는 순간부터 매칭된다.
  그래서 `s2_lookback_days: 0`으로 lookback을 끄고 `year`에 시간 범위를 맡기는 것을 권장한다.

### arXiv (보조 소스, 프리프린트)

- `export.arxiv.org/api/query`, `submittedDate` 내림차순, `arxiv_lookback_days` 안의 제출분만. venue 필터가 없으므로 결과는 반드시 relevance 필터를 거친다.
- `arxiv` 라이브러리 내장 재시도: 3초 간격 최대 3회(총 4회). 실패하면 그 topic의 arXiv만 건너뛴다.
- 쿼리는 `cat:` 카테고리와 `abs:"구문"`을 AND/OR로 조합하는 arXiv 검색 문법.
- **현황(2026-09-14)**: arXiv 검색 API가 간헐적 `429 Rate exceeded`/503을 낸다. 9/12 4토픽 성공 → 9/13 2토픽 → 9/14 0토픽으로
  같은 시각·같은 요청량에서 단계적으로 악화됐고, 하루 중 열리고 닫히는 시간대가 있다. arXiv 운영자는 "429는 서버 용량 문제"라고 밝혔으며,
  한 실행 안에서 앞 요청은 통과하고 뒤 요청이 막히는 패턴은 IP별 예산이 매우 작다는 뜻이라 학교 공유 IP의 다른 트래픽이 겹치는 것으로 보인다.
  (학교 밖 네트워크에서 같은 요청이 200이면 IP 요인 확정.) 막힌 날은 arXiv 후보가 0이 되고 pending에 있던 arXiv 논문만 발행된다.

---

## 6. relevance 필터

topic별로 묶어 처리하며, 탈락은 `filtered_out`으로 기록해 다시 채점하지 않는다.

**1단계 — 임베딩 프리필터 (CPU, 밀리초).** `interest` 문단과 `title + abstract`를
`all-MiniLM-L6-v2`로 임베딩해 cosine이 `embedding_threshold` 미만이면 컷.
abstract가 없는 논문은 제목만으로 점수를 내되 하드컷하지 않고 judge로 넘긴다(title-only 유사도는 체계적으로 낮다).
실측 예: 정면 관련 0.5~0.7, 주변부 0.2~0.3, 무관 0.2 이하.

**2단계 — LLM judge (`llm_judge: true`).** 관심사·제목·출처·abstract를 주고 0~10 정수 점수와 한 문장 근거를 받는다.

- thinking 꺼짐 + `response_format`(JSON 스키마 `{"score": int 0~10, "reason": str}`) 서버 강제 → 파싱 실패가 구조적으로 없다.
- temperature 0, 논문당 약 5초. `judge_threshold`(6) 미만이면 `filtered_out`에 근거 기록.
- 호출 자체가 3회 실패하면 보수적으로 통과(놓치는 것보다 낫다). 점수 없이 통과한 논문은 DB `judge_score`가 NULL.

채점 기준(프롬프트): 9~10 핵심 주제 정면 / 6~8 상당히 관련 / 3~5 주변부 / 0~2 무관·명시적 제외.
감을 잡을 때는 `--skip-filter`로 전부 요약해 보고 로그의 `sim=`·judge 점수와 비교한다.

---

## 7. 발행 쿼터와 pending 대기열

`quota.s2_per_topic` / `quota.arxiv_per_topic`은 실행(=하루)당 채널·소스별 발행 상한이다.

```
후보 = [pending 대기열 (topic×소스별 오래된 순)] + [신규 필터 통과분 (최신순)]
       └── apply_quota(): 입력 순서대로 (topic, source)별 상한까지 선택
선택됨            → ⑦ 처리 → done
선택 안 된 신규    → store.mark_pending()  (Paper 전체를 payload JSON으로 보존)
선택 안 된 대기열  → 그대로 유지 (진입 시각 보존 → FIFO 순서 유지)
```

- 대기열 논문은 재검색·재채점 없이 payload에서 복원해 바로 요약·발행한다. 검색 결과에 다시 나타나도 신규로 잡히지 않는다.
- `pending_max_age_days`(30)를 넘긴 논문은 `expired`로 폐기된다.
- `--limit`에 걸린 신규분도 pending으로 간다(누락 없음).
- 현재 순서는 "judge → 쿼터"다. 후보가 많아져 judge 비용이 커지면 "임베딩 → 쿼터 → 선택분만 judge(+탈락 보충)"로 바꿀 여지가 있다.

확인:
```bash
sqlite3 ~/data2nas/Paper/papers.db "select topic, source, count(*) from papers where status='pending' group by 1,2;"
```

---

## 8. PDF 확보·추출·요약

### PDF 확보 (폴백 체인)

출판사 서버(ACM/IEEE 등)는 봇 요청을 403으로 막는 경우가 많다. 앞 단계가 전부 실패했을 때만 다음으로 간다.

1. 검색 API가 준 직접 링크 (S2 `openAccessPdf` / arXiv PDF)
2. arXiv 미러 (arXiv id가 있을 때)
3. **Unpaywall** — DOI로 합법 OA 사본 조회 (`fetch.unpaywall_email` 필요)
4. **arXiv 제목 검색** — 정규화 제목 완전 일치 + 1저자 성 확인
5. 전부 실패 → abstract 기반 요약 (`(abstract 기반 요약)` 표기)
6. abstract도 없음 → 발행하지 않음. `failed` 기록, 다음 실행 1회 재시도 후 `failed_permanent`

### 추출

- pymupdf로 전체 텍스트 → References/Bibliography 헤딩 이후 제거(문서 30% 지점 이후, 라인 전체가 헤딩일 때만) → 빈 줄 정리.
- `max_input_chars`(150k) 초과 시 앞 70% + 뒤 30%만 남기고 중간 절단(안전망, 실제로는 드묾).
- **figure**: 앞 8페이지에서 `Figure N` 캡션을 기준점으로, 캡션 위 같은 단의 드로잉·이미지·짧은 레이블을 tight bbox로 묶고 캡션을 붙여 렌더(2x). 표·정리 박스·옆 단 병합·헤더 유입을 캡션 기준으로 배제한다. `slack.max_figures`개까지.

### 요약

- 시스템 프롬프트가 6개 섹션을 고정한다: **한 줄 요약 / 문제 정의 / 핵심 방법 / 기존 연구 대비 novelty / 실험 결과 / 한계점**.
- 한국어(`llm.language`), temperature 0.3, thinking 꺼짐, 출력 상한 8192 토큰. `<think>` 태그가 남아 있으면 제거한다.
- 실측: 전문 40~46k자 입력에 100~150초, 출력 3~4.5k자.

---

## 9. 메신저 발행

`publisher.backend`로 선택한다. 둘 다 없어도 `dry_run: true` 또는 `--dry-run`으로 전 과정을 테스트할 수 있다.

### Slack (현재)

포스팅 구조: **채널 메인 메시지**(제목·저자·venue·연도·인용수·링크·한 줄 요약·`#topic`) + **스레드 댓글**(6섹션 전체 요약) + **figure 첨부**(스레드에 댓글 하나).

봇 토큰: https://api.slack.com/apps → Create New App → OAuth & Permissions에 `chat:write`(+ figure용 `files:write`) → Install → `xoxb-...` 토큰을 `~/.paperbot_env`에. 대상 채널마다 `/invite @paperbot`.

topic별 `channel`을 지정하면 주제마다 다른 채널로 간다(미지정 topic은 `slack.channel`). 채널명 또는 채널 ID 모두 가능.

| 오류 | 원인 |
|---|---|
| `not_in_channel` | 공개 채널에 봇 미초대 |
| `channel_not_found` | 채널명 오타 또는 비공개 채널에 봇 미초대(초대 전엔 보이지 않음) |
| `invalid_auth` / `missing_scope` | 토큰 오류 / 스코프 누락 |

이런 **설정 오류는 실행 전체를 즉시 중단**하며 논문을 failed로 소모하지 않는다. 고치고 재실행하면 이어서 처리된다.

### Zulip

`zuliprc`를 프로젝트 루트에 두고(`.gitignore`) `publisher.backend: "zulip"`. 스트림에 논문당 topic 하나(제목 60자), 전체 요약 한 메시지.
topic의 `channel`은 스트림 이름으로 해석된다.

---

## 10. 실행과 테스트

```bash
python run.py                                    # 전체 파이프라인
python run.py --dry-run --topic graph-transformer --limit 1   # 한 topic 1편, 전송 없이
python run.py --backfill --dry-run               # 첫 실행: arXiv lookback 무시
python run.py --skip-filter --dry-run --limit 2  # 필터 없이 (threshold 튜닝용)
```

| 옵션 | 설명 |
|---|---|
| `--config PATH` | config.yaml 경로 (기본: run.py 옆) |
| `--dry-run` | 전송 대신 stdout 출력 (DB 기록은 그대로 됨) |
| `--limit N` | 요약·발행할 논문 수 제한. 잘린 신규분은 pending으로 |
| `--topic NAME` | 특정 topic만 |
| `--backfill` | arXiv lookback 해제 |
| `--skip-filter` | relevance 필터 생략 (디버그) |

> **dry-run도 DB에 done을 기록한다.** 운영 DB를 건드리지 않고 테스트하려면 DB를 복사하고
> `storage.db_path`만 바꾼 임시 config를 `--config`로 넘긴다.

- **오프라인 회귀 테스트**: `python smoke_test.py` — 네트워크·Ollama 불필요. 쿼터 항목 4개는 config가 3/2일 것을 가정해 현재(2/1) 설정에서 FAIL이 정상.
- **채널 실전송 점검**: `scripts/test_send.sh --check`(토큰·스코프·Ollama·채널 확인만) / `scripts/test_send.sh`(topic마다 1편 실제 전송).

---

## 11. 주기 실행

현재 방식은 **tmux 러너**다. crontab에는 재부팅 복구용 `@reboot` 두 줄만 있다.

```cron
@reboot tmux new-session -d -s ollama 'CUDA_VISIBLE_DEVICES=1,2 GGML_VK_VISIBLE_DEVICES=1,2 OLLAMA_CONTEXT_LENGTH=65536 $HOME/ollama/bin/ollama serve 2>&1 | tee -a $HOME/ollama/serve.log'
@reboot tmux new-session -d -s paperbot '$HOME/paperbot/scripts/loop.sh 07:00'
```

`scripts/loop.sh HH:MM`은 지정 시각까지 잠들다가 `~/.paperbot_env`를 읽고, Ollama가 죽어 있으면 재시작한 뒤 `run.py`를 실행하고 `logs/cron.log`에 남긴다.
실행 시각을 바꾸려면 crontab의 인자와 tmux 세션(`tmux kill-session -t paperbot` 후 재생성)을 함께 바꾼다.
config·코드 수정은 재시작이 필요 없다(매일 새 프로세스).

```bash
tmux attach -t paperbot        # 상태 확인 (Ctrl+b d 로 분리)
```

cron으로 직접 돌리려면 `0 7 * * * . $HOME/.paperbot_env && cd $HOME/paperbot && ./venv/bin/python run.py >> logs/cron.log 2>&1` 한 줄로 대체할 수 있다. 두 방식을 병행하면 lockfile이 중복을 막지만 하나만 쓴다.

### GPU 점유

| 상태 | VRAM |
|---|---|
| 대기 (ollama serve만) | 0 MiB |
| 봇 실행 중 | 약 21GB (GPU 1·2, 모델 15GB + 64K KV 4GB + 버퍼) |
| 실행 종료 직후 | 0 MiB — 봇이 종료 시 모델을 명시적으로 언로드 |

---

## 12. 데이터·로그·상태

| 항목 | 위치 |
|---|---|
| 상태 DB | `storage.db_path` (현재 NAS `/home/gwlee/data2nas/Paper/papers.db`) |
| PDF 임시 저장 | `storage.pdf_dir` (요약 후 삭제, `keep_pdfs: true`면 보존) |
| lockfile | DB 폴더의 `.lock` (죽은 락은 자동 회수) |
| 실행 로그 | `logs/paperbot.log` (5MB×3 회전), 실행 출력 `logs/cron.log` |

### `papers` 테이블 status

| status | 의미 | 다음 실행에서 |
|---|---|---|
| `done` | 요약·발행 완료 | 건너뜀 |
| `filtered_out` | relevance 필터 탈락 (`judge_score`, 사유 기록) | 건너뜀 (재채점 안 함) |
| `pending` | 필터 통과, 쿼터 대기. `payload`에 Paper JSON | 대기열에서 FIFO로 우선 발행 |
| `expired` | pending이 `pending_max_age_days` 초과 | 건너뜀 |
| `failed` | 1회 실패 | 다시 검색되면 재시도 |
| `failed_permanent` | 재시도도 실패 | 건너뜀 |

`payload` 컬럼은 첫 실행 시 자동 추가된다(additive 마이그레이션). 변경 전 백업: `papers.db.bak-20260914`.

---

## 13. 운영 메모

- **arXiv API 429/503**: 5장 참고. 요청량(하루 4건, 3초 간격, 10편)은 이미 최소라 코드로 풀 문제가 아니다. 학교 밖에서 확인 테스트 후
  실행 후반(요약 종료 뒤) 재시도·하루 2회 실행·arXiv 몫을 S2로 보충 중 선택 예정. 현재는 `arxiv` 라이브러리 기본 재시도(3초×3) 그대로.
- **S2 429**: 실행 초반에 몇 초 단위로 발생하며 3·5·10초 백오프로 대부분 회복된다. 4회 모두 실패하면 그 topic의 S2만 빠지고, 대기열에 있던 논문은 그대로 발행된다.
- **judge 실패 이력**: 이전(qwen3.5, thinking on) 구성에서는 thinking이 토큰 상한을 소진해 빈 응답 → 3회 동일 실패 → 미채점 통과가 발생했다. thinking 끄기 + JSON 스키마 강제로 해결되었고, 실패는 응답 시간이 상한(약 100초)에 일정하게 걸리는 패턴으로 구분할 수 있다.
- **중복 발행**: uid 정규화로 소스 간 중복은 합쳐진다. S2가 arXiv id를 누락한 학회 논문과 같은 프리프린트가 각각 들어오는 극히 드문 경우만 두 번 나갈 수 있다(허용).
- **향후 확장**: `publisher.py`의 `NotionPublisher` 스텁에 API 연동을 채우고 `make_publisher`에 연결하면 Notion 아카이브가 가능하다.
