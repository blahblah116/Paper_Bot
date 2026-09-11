# paperbot

> Local-LLM paper summary bot — fetches new papers (Semantic Scholar + arXiv),
> filters by relevance, summarizes with Ollama, posts to Slack/Zulip.

주제(topic)와 학회(venue) 필터를 조합해 논문을 주기적으로 수집하고, 로컬
LLM(Ollama)으로 구조화된 요약을 생성해 **Slack 또는 Zulip**에 포스팅하는 봇.
외부 LLM API 없이 연구실 GPU 서버에서 전부 돌아간다.

## 동작 개요

1. **검색** — topic별로 두 소스에서 수집:
   - **Semantic Scholar** (주 소스): venue/연도/최소 인용 수 필터 공식 지원
   - **arXiv** (보조 소스): 학회 게재 전 최신 프리프린트
2. **dedupe** — DOI/arXiv id 기반 uid로 소스 간 중복 통합, sqlite로 처리 이력 관리
3. **2단계 relevance 필터** — ① 임베딩 프리필터(CPU, cosine) ② LLM judge(0~10 채점)
4. **요약** — PDF 텍스트 추출(References 제거, 길이 절단) → 6개 섹션 구조 요약
5. **포스팅** — `publisher.backend`로 선택:
   - **Slack** (기본): 채널에 메인 메시지(제목·메타·링크·한 줄 요약) + 스레드에 전체 요약
   - **Zulip**: 스트림에 논문당 topic 하나, 전체 요약 한 메시지
   (탈락 논문은 `filtered_out`으로 기록되어 재채점하지 않음)

## 요구 사항

- Python 3.10+, 상시 실행 중인 Ollama (`http://localhost:11434/v1`)
- (권장) Semantic Scholar API 키 — 환경변수 `S2_API_KEY`
- (실제 전송 시) Slack 봇 토큰(`SLACK_BOT_TOKEN`) 또는 Zulip `zuliprc`

### V100(Volta) 관련 메모

- bf16 미지원 → fp16만. FlashAttention-2 / AWQ / Marlin 커널 미지원 →
  LLM 서빙은 **Ollama(GGUF)**, vLLM은 쓰지 않는다.
- 이 서버의 NVIDIA 드라이버(535)가 Ollama 0.34의 CUDA 백엔드 요구(550+)보다
  오래되어 **Vulkan 백엔드로 동작 중**이다. 드라이버를 550+로 올리면 CUDA 백엔드가
  활성화되어 prefill이 더 빨라진다 (관리자 문의).
- 임베딩 모델은 CPU에서 실행 (GPU는 요약 모델 전용).

## 설치

```bash
cd /path/to/paperbot
python3 -m venv venv
source venv/bin/activate
# torch는 CPU 휠을 먼저 설치 (CUDA 휠 ~3GB 회피, 임베딩은 CPU로 충분)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 설정 파일 생성 (실제 config.yaml은 gitignore되어 커밋되지 않음)
cp config.example.yaml config.yaml
# → topics의 주제·채널, unpaywall_email 등을 채운다
```

## Ollama 설치·실행 (sudo 없이, 유저 홈)

```bash
# 최신 버전은 .tar.zst 포맷
curl -fL https://ollama.com/download/ollama-linux-amd64.tar.zst -o /tmp/ollama-linux-amd64.tar.zst
mkdir -p ~/ollama && tar --zstd -xf /tmp/ollama-linux-amd64.tar.zst -C ~/ollama
echo 'export PATH="$HOME/ollama/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc
```

tmux로 상시 실행. **환경변수 3개가 중요하다**:

```bash
tmux new-session -d -s ollama \
  'CUDA_VISIBLE_DEVICES=1,2 GGML_VK_VISIBLE_DEVICES=1,2 OLLAMA_CONTEXT_LENGTH=65536 \
   $HOME/ollama/bin/ollama serve 2>&1 | tee -a $HOME/ollama/serve.log'
```

| 변수 | 왜 필요한가 |
|------|-------------|
| `OLLAMA_CONTEXT_LENGTH=65536` | 미설정 시 Ollama가 **전체 VRAM 기준으로 컨텍스트를 자동 설정**(이 서버에선 262144)해서 KV 캐시가 GPU 3장에 27GB로 분산된다. 64K는 27b 기준 GPU 2장에 ~22GB로 수렴. |
| `GGML_VK_VISIBLE_DEVICES=1,2` | Vulkan 백엔드의 GPU 고정 (공용 서버 매너). Vulkan은 `CUDA_VISIBLE_DEVICES`를 무시한다. |
| `CUDA_VISIBLE_DEVICES=1,2` | 드라이버 업데이트로 CUDA 백엔드가 활성화될 경우를 대비. |

재부팅 후 자동 시작하려면 crontab에:

```cron
@reboot tmux new-session -d -s ollama 'CUDA_VISIBLE_DEVICES=1,2 GGML_VK_VISIBLE_DEVICES=1,2 OLLAMA_CONTEXT_LENGTH=65536 $HOME/ollama/bin/ollama serve 2>&1 | tee -a $HOME/ollama/serve.log'
```

> **num_ctx 주의**: 봇은 매 요청에 `num_ctx`를 extra_body로 명시하지만, 실측 결과
> Ollama 0.34의 OpenAI 호환 엔드포인트는 이를 **무시**한다. 실효 설정은 서버의
> `OLLAMA_CONTEXT_LENGTH`뿐이다. 봇이 첫 호출 후 `/api/ps`로 실제 컨텍스트를
> 확인해 config의 `llm.num_ctx`보다 작으면 경고 로그를 남긴다.

### 모델 준비 (V100 16GB × 2장 분산)

```bash
ollama pull qwen3.5:27b    # 17GB Q4, 기본값 — 64K 컨텍스트 포함 GPU 2장에 ~22GB
```

- 실측 (V100 ×2, Vulkan): 로드 11초, prefill 415 tok/s, 생성 25.5 tok/s
  → 논문 전문(15k+ 토큰) 요약 1편에 2~4분.
- 경량 대안: `qwen3.5:9b` (6.6GB, GPU 1장 — 속도 우선일 때. GGML_VK_VISIBLE_DEVICES를
  한 장으로 줄이고 config `llm.model`만 교체).
- qwen 계열의 thinking 토큰(`<think>…</think>`)은 봇이 자동 제거한다.

## Semantic Scholar API 키 (권장)

- 무키로도 동작하지만 공용 rate limit이라 429가 잦다 (봇이 자동 백오프하지만 느림).
- https://www.semanticscholar.org/product/api 에서 무료 키를 신청한다.
- 키는 **config에 넣지 않고** 환경변수로 주입한다:

```bash
export S2_API_KEY="발급받은키"        # 쉘 테스트용
# cron에서는 crontab 상단에:  S2_API_KEY=발급받은키
```

## 설정 (config.yaml)

주제·학회·연도·인용 수·필터 강도는 전부 config에서 조절한다 (코드 수정 없이).

```yaml
topics:
  - name: "graph-transformer"        # Zulip 태그(#name)로 표시
    interest: >                       # relevance 필터(임베딩·judge)의 기준 서술
      Graph Transformer 및 ... 단순 GNN 응용 논문은 제외.
    sources:
      semantic_scholar:
        query: '"graph transformer" | "graph mamba"'   # S2 bulk 검색 문법
        venues: ["ICLR", "NeurIPS", "ICML", "KDD"]     # 빈 리스트 = venue 필터 없음
        year: "2024-"                                  # "2024" | "2024-2026" | "2024-"
        min_citations: 0
      arxiv:
        enabled: true
        query: 'cat:cs.LG AND (abs:"graph transformer" ...)'  # arXiv 검색 문법
```

### venue 이름 확인 방법

S2의 venue 필터는 요청 시 **주요 학회 축약형(ICLR, NeurIPS, ICML, KDD, CVPR 등)을
알아서 매칭**해 준다 (실측 확인). 단, 응답의 `venue` 필드는 전체 이름
("International Conference on Learning Representations")으로 돌아오며, 마이너한
학회/워크숍은 축약형이 안 먹을 수 있다. 확인 방법:

```bash
curl -s "https://api.semanticscholar.org/graph/v1/paper/search/bulk?query=YOUR_QUERY&venue=VENUE_NAME&fields=title,venue" | head -40
```

결과가 0건이면 venue 표기를 바꿔서 (전체 이름 ↔ 축약형) 다시 시도하고,
응답에 찍히는 `venue` 문자열을 그대로 쓰는 것이 가장 확실하다.

### 필터 튜닝

- `filter.embedding_threshold` (기본 0.35): interest↔abstract cosine 하한.
  실측 예 — 정면 관련 0.67 / 주변부 0.23 / 무관 0.19. 너무 많이 잘리면 0.30으로.
- `filter.judge_threshold` (기본 6): LLM judge 0~10 점수 하한.
  실측 예 — 정면 관련 9 / interest에서 제외한 부류 2 / 인접 분야 4.
- 감 잡기: `--skip-filter`로 전부 요약해 보고, 로그의 sim/judge 점수와 비교.
- abstract가 없는 논문은 임베딩 하드컷 없이 judge로 넘어간다 (title-only 유사도는
  체계적으로 낮게 나와 같은 threshold로 자르면 과잉 컷이기 때문).

## 메신저 설정

`config.yaml`의 `publisher.backend`로 선택한다 (`"slack"` 또는 `"zulip"`).
**둘 다 없어도** `dry_run: true` 또는 `--dry-run`으로 전 과정 테스트 가능.

### Slack (기본)

포스팅 구조: 채널 메인 메시지(제목·저자·venue·인용·링크·한 줄 요약) + 스레드 댓글(6섹션 전체 요약).
채널은 논문 피드처럼 훑고, 관심 논문만 스레드를 열어 읽는 구조다.

봇 토큰 발급 (워크스페이스 관리자 권한 필요할 수 있음):

1. https://api.slack.com/apps → **Create New App** → From scratch → 이름(예: paperbot)·워크스페이스 선택
2. **OAuth & Permissions** → Bot Token Scopes에 **`chat:write`** 추가
3. **Install to Workspace** → 승인 → **Bot User OAuth Token**(`xoxb-...`) 복사
4. Slack에서 대상 채널에 봇 초대: `/invite @paperbot`
5. 토큰을 환경변수로 주입 (config에 넣지 않는다):

```bash
export SLACK_BOT_TOKEN="xoxb-..."     # 쉘 테스트용
# cron에서는 crontab 상단에:  SLACK_BOT_TOKEN=xoxb-...
```

- `slack.channel`은 채널명(`"#papers"`) 또는 채널 ID 모두 가능.
- 전송 실패 시 흔한 원인: `not_in_channel`(공개 채널에 봇 미초대),
  `channel_not_found`(채널명 오타 **또는 비공개 채널에 봇 미초대** — 비공개 채널은
  초대 전엔 봇에게 보이지 않는다), `invalid_auth`(토큰 오류), `missing_scope`(chat:write 누락).
- 이런 **설정 오류는 실행 전체를 즉시 중단**하며 논문을 failed로 소모하지 않는다 —
  설정을 고치고 다시 실행하면 그대로 이어서 처리된다.

### topic별 채널 라우팅

topic마다 `channel`을 지정하면 각 주제의 논문이 해당 채널로 간다
(미지정 topic은 `slack.channel` 기본값, Zulip backend에서는 스트림 이름으로 해석):

```yaml
topics:
  - name: "graph-transformer"
    channel: "#papers-gt"              # 이 topic 전용 채널
    ...
  - name: "cad-reverse-engineering"
    channel: "#papers-cad"
    ...

slack:
  channel: "#papers"                  # channel 미지정 topic의 기본값
```

**모든 채널에 봇을 초대해야 한다** (`/invite @paperbot`).

### figure 첨부 (`slack.attach_figure`)

논문 PDF에서 대표 figure를 최대 `slack.max_figures`개(기본 3) 뽑아 요약 스레드에
댓글 하나로 첨부한다. **files:write 스코프** 필요 (없으면 경고 1회 후 텍스트만 발행).

선별 휴리스틱 — 앞 8페이지에서:
1. 임베디드 래스터 이미지 + **벡터 다이어그램**(drawing cluster — ML 논문의
   method/아키텍처 그림은 벡터가 많음) 후보 수집
2. 크기·가로세로비·색 다양성·압축 밀도로 로고/배너/그라디언트 장식 컷
3. 면적 상위 N개를 읽기 순서(페이지→위치)로 첨부. 아무것도 없으면 1페이지 렌더 1장

"메인 method figure"를 의미로 이해하는 건 아니고 크기 기반 휴리스틱이다 —
실측으론 보통 teaser·아키텍처·주요 결과가 잡힌다.

### PDF 확보 전략 (폴백 체인)

출판사 서버(ACM/IEEE 등)는 봇 요청을 403으로 차단하는 경우가 많다. 봇은 아래
순서로 시도하며, 조회형 후보(3·4)는 앞 단계가 전부 실패했을 때만 실행된다:

1. 검색 API가 준 직접 링크 (S2 `openAccessPdf` / arXiv PDF)
2. arXiv 미러 (arXiv id가 있을 때)
3. **Unpaywall** — DOI로 합법 OA 사본(리포지토리·저자 원고) 조회.
   `fetch.unpaywall_email` 설정 필요 (비우면 이 단계 생략)
4. **arXiv 제목 검색** — S2가 arXiv 링크를 누락한 프리프린트 발견
   (정규화 제목 완전 일치 + 1저자 성 확인으로 오매칭 방지)
5. 전부 실패: abstract 기반 요약 (`(abstract 기반 요약)` 표기)
6. abstract도 없으면: 발행하지 않음 (failed 기록 → 다음 실행 1회 재시도 후 영구 skip)

### Zulip

Zulip 관리자에게 봇 계정 생성을 요청해 `zuliprc`를 받아 프로젝트 루트에 저장한다
(`.gitignore` 처리됨). 봇을 대상 스트림에 구독시키고 `publisher.backend: "zulip"`으로 변경:

```ini
[api]
email=paperbot-bot@your-zulip.example.com
key=BOT_API_KEY
site=https://your-zulip.example.com
```

포스팅 구조: 스트림에 논문당 topic 하나(제목 60자), 전체 요약 한 메시지.

## 실행

```bash
python run.py                                  # 전체 파이프라인
python run.py --dry-run --limit 3              # 전송 없이 3편만 (테스트)
python run.py --topic graph-transformer        # 특정 topic만
python run.py --backfill --dry-run             # 첫 실행: lookback 무시하고 과거분 포함
python run.py --skip-filter --dry-run --limit 2  # 필터 없이 (threshold 튜닝용)
```

| 옵션 | 설명 |
|------|------|
| `--config PATH` | config.yaml 경로 (기본: run.py 옆) |
| `--dry-run` | config 무시하고 stdout 출력 |
| `--limit N` | 요약할 논문 수 제한 |
| `--topic NAME` | 특정 topic만 |
| `--backfill` | lookback 필터 해제 (첫 실행/과거분 수집) |
| `--skip-filter` | relevance 필터 생략 (디버그) |

> **`--limit` 주의**: limit은 "요약·발행 수"만 제한한다. relevance 필터(LLM judge,
> 논문당 ~30초)는 미처리분 전체에 돌므로, 빠른 테스트에는 `--topic` +
> `--skip-filter`를 함께 쓰는 것이 좋다. judge를 통과했지만 limit에 걸려 발행되지
> 않은 논문은 DB에 기록되지 않아 다음 실행 때 다시 처리된다(누락 없음).

### 소스별 신선도 특성 (운영 참고)

- **arXiv**: 제출 후 1~2일 내 검색됨 — 신선도 담당. `arxiv_lookback_days`로 조절.
- **S2 (venue 필터)**: proceedings 인덱싱이 **수개월 늦고 버스트로** 들어온다.
  또한 `min_citations`를 쓰면 논문이 인용 문턱을 넘는 순간부터 쿼리에 매칭되므로,
  짧은 lookback은 "이제 막 저명해진" 논문을 놓친다.
  **권장: `s2_lookback_days: 0`** — lookback을 끄고 시간 범위를 topic별 `year`
  필터에 위임한다. 물량은 `최신순 정렬 + max_results + DB dedupe`가 자연히 제한하고,
  인용 문턱을 늦게 넘은 논문도 (year 범위 내라면) 연식과 무관하게 잡힌다.
  arXiv는 year 같은 서버측 시간 필터가 없으므로 `arxiv_lookback_days`(>=1)가 필수.
- S2 bulk 검색은 `publicationDate` 내림차순으로 요청하며 lookback에 도달하면
  페이지네이션을 조기 중단한다 (무의미한 API 호출 없음).

## 주기 실행 (둘 중 하나 선택)

두 방식 모두 토큰은 `~/.paperbot_env`(권한 600)에서 읽는다:

```bash
# ~/.paperbot_env
export SLACK_BOT_TOKEN="xoxb-..."
export S2_API_KEY="..."        # 있으면
```

### 방식 A — cron (이 서버에 등록되어 있음)

```cron
@reboot tmux new-session -d -s ollama 'CUDA_VISIBLE_DEVICES=1,2 GGML_VK_VISIBLE_DEVICES=1,2 OLLAMA_CONTEXT_LENGTH=65536 $HOME/ollama/bin/ollama serve 2>&1 | tee -a $HOME/ollama/serve.log'
0 9 * * * . $HOME/.paperbot_env && cd $HOME/paperbot && ./venv/bin/python run.py >> logs/cron.log 2>&1
```

확인: `crontab -l`, 시간 변경: `crontab -e`.

### 방식 B — tmux 러너 (`scripts/loop.sh`)

cron 없이 tmux 세션 하나로 상주시키는 방식. 지정 시각까지 잠들어 있다가 실행:

```bash
tmux new -s paperbot '~/paperbot/scripts/loop.sh 09:00'
tmux attach -t paperbot     # 상태 확인 (Ctrl+b d 로 분리)
```

두 방식을 병행해도 lockfile이 중복 실행을 막지만, 하나만 쓸 것 (A를 쓰면
`crontab -e`에서 9시 라인 유지, B를 쓰면 그 라인 삭제).

### GPU 점유 타임라인 ("특정 시간만 점유")

| 상태 | VRAM 점유 |
|------|-----------|
| 대기 중 (ollama serve만 떠 있음) | **0 MiB** — 모델 미로드 시 serve는 GPU를 안 씀 (실측) |
| 봇 실행 중 | ~7.3GB (GPU 0 한 장, 모델+32K KV) |
| 실행 종료 직후 | **즉시 0 MiB** — 봇이 종료 시 모델을 명시적으로 언로드 |

즉 ollama serve를 24시간 띄워놔도 실제 GPU 점유는 봇이 도는 시간(하루 수십 분)뿐이다.
(봇이 비정상 종료해 언로드를 못 해도 `OLLAMA_KEEP_ALIVE` 기본 5분 후 자동 언로드.)

- 중복 실행은 `data/.lock`으로 방지 (죽은 락은 자동 회수).
- 로그: `logs/paperbot.log` (5MB×3 회전), 실행 출력은 `logs/cron.log`.
- 발행 설정 오류(토큰 없음/채널 없음 등)는 fetch 전 또는 발생 즉시 실행을 중단하며
  논문을 소모하지 않는다 — 설정을 고치고 재실행하면 이어서 처리된다.

## 데이터/상태 (data/papers.db)

| status | 의미 |
|--------|------|
| `done` | 요약·포스팅 완료 |
| `filtered_out` | relevance 필터 탈락 (judge_score 기록, 재채점 안 함) |
| `failed` | 1회 실패 — 다음 실행 때 재시도 |
| `failed_permanent` | 재시도도 실패 — 영구 skip |

## 향후 확장 (이번 범위 밖)

- **Notion 아카이브**: `paperbot/publisher.py`의 `NotionPublisher` 스텁에 API 연동을
  채우고 `make_publisher`에 연결.
