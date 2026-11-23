# sft-data-pipeline (RAFT)

RAG 用途の SFT データセットを **RAFT (Retrieval-Augmented Fine-Tuning)** 方式で構築するパイプライン。
チャンク済みドキュメントから「質問 + 検索文脈（ゴールデン+ディストラクタ）+ 引用付き CoT 回答」を生成し、
`継続事前学習` の SFT トレーナーが消費する `messages` 形式 JSONL を出力する。

隣の `cpt-data-pipeline` の規約（`llm_factory.py` のプロバイダ抽象、pydantic-ai 構造化出力、
非同期+セマフォのバルク処理、フェーズ別 checkpoint）を踏襲している。

## RAFT の流れ

```
corpus.jsonl
  → Phase1 質問生成      （チャンクごとに回答可能な質問を生成）
  → Phase2 CoT回答生成   （オラクル文書を根拠に ##begin_quote## 引用付き CoT）
  → Phase3 RAFT合成      （確率 P でゴールデン+ディストラクタ、(1-P) でディストラクタのみ）
  → Phase4 エクスポート  （Swallow トークナイザで max_len フィルタ → messages JSONL）
```

各フェーズ間は `data/.checkpoint/` に中間 JSONL を残し、`--resume` で再開できる。
Phase1/2 は LLM 呼び出し、Phase3/4 は決定的・安価なので常に再実行される。

## タスク（`--task`）

同じ RAFT 機構（検索文脈＋ディストラクタ＋oracleドロップアウト＋引用付きCoT）を、
Phase1/2 のプロンプトを差し替えて別タスクに特化できる。

| `--task` | Phase1（クエリ生成） | Phase2（回答） | 用途 |
|---|---|---|---|
| `qa`（既定） | チャンクから**質問**を生成 | 質問に引用付きCoTで**回答** | RAG向けQA |
| `regulation` | 規程条文から**その規程が適用される工程/作業**を逆生成 | 工程に規程が該当するかを引用付きCoTで**判定** | 「工程→該当規定の判定」特化 |

```powershell
# 工程→該当規定 判定モデル用データ（正解ラベル不要・規程から工程を逆生成してブートストラップ）
python scripts/build_raft.py corpus_clean.jsonl -o data/output/sft_regulation.jsonl `
  --task regulation --max-doc-chars 450 --num-distractors 4 --questions-per-chunk 1 `
  --system-prompt regul --system-dropout 0.15 --val-split 0.1 --keep-intermediate
```
`regulation` では文脈末尾のラベルが「質問:」→「工程:」に変わり、ディストラクタ（無関係規程）の
中から該当規程を選び根拠引用する挙動を学習する。

## 拒否例（`--absent-mode refuse` / `--twin-ratio`）

既定の合成（RAFT 論文どおり）では、確率 (1 − P) で正解の文書を文脈から抜いても、答えは正解の文書を見て書いた
引用付き回答のまま残る。「資料に答えが無ければ断る」ことを教えたい場合、これらの例は逆の手本になる。
次の 2 つのオプションで拒否例を作る（`--task qa` のみ）。

| オプション | 作る例 |
|---|---|
| `--absent-mode refuse` | ゴールデンを抜いた例の答えを拒否文に置き換える（その質問の回答例は作らない） |
| `--twin-ratio R` | 回答例のうち割合 R について、同じ質問・同じ文書の並びで正解の文書だけを差し替えた拒否例（双子）を足す |

拒否例と回答例が形で見分けられないようにするため、次の処理を行う。

- 文書数・並び・書式は回答例と同じにする。正解の文書の位置に、質問に字面が近く長さも近い別の文書を入れる
- 差し替え後の文書群から模範解答が導けないことを LLM で判定する。導ける場合は次の候補を試し、候補 3 件とも導ける例は捨てる
- 拒否文は回答例と同じ骨格（主題の確認 → 資料の確認 → 最終的な回答）で LLM に書かせ、長さ・引用マーカーの有無・
  最終回答行が拒否であること・模範解答の漏れを機械で検証する。3 回落ちた例は捨てる（元の回答には戻さない）
- 双子は文書数ごとに同じ割合で作る
- train/val は質問単位で分ける（同じ質問の回答例と双子が両側に分かれない）

拒否例は `data/.checkpoint/{stem}_refusals.jsonl` に 1 例 1 行で残り、`--resume` で使い回す。
`--dry-run` では LLM を呼ばず、判定を省いて決まった拒否文を入れる。

学習側の入力形式に合わせるオプション:

- `--context-format bullets` … 文脈を `<DOCUMENT>` タグでなく「参考資料:」の見出しと箇条書きにする
- `--user-instruction rag_refuse` … user の先頭に指示文（逐語引用・最終回答行・資料に無ければ断る）を置く。
  任意の文字列も指定できる。推論・評価で使う指示文と一字一句そろえること

```bash
python scripts/build_raft.py data/input/corpus.jsonl -o data/output/sft_refuse.jsonl \
  --max-doc-chars 450 --absent-mode refuse --twin-ratio 0.25 \
  --context-format bullets --user-instruction rag_refuse --val-split 0.1 --keep-intermediate
```

出力の隣に `{stem}_manifest.json`（生成モデル・引数・入力の md5・種類ごとの件数と形）を書く。

## ディストラクタを検索の上位から取る（`--distractor-mode retrieved`）

既定（`random`）はコーパス全体からの無作為抽出で、RAFT 論文の公式実装と同じである。無作為の文書は
正解と別の文書になることが多く、推論時に検索が返す「質問に似ているが答えは載っていない文書」とは形が違う。
`retrieved` は、質問ごとに実際に検索して、その上位から文脈を作る。

```
Phase2 の回答
  → 検索          BM25（文字 2・3-gram）と埋め込みで各 50 件 → 順位融合 → 上位 20 件をリランカーで並べ替え
  → 質問の判定    （--question-check on）自動採用 / 要確認 / 自動除外 と理由
  → Phase3        文脈 = 検索の上位 (k+1) 件。正解の文書が入れば回答例、入らなければゴールデン抜きの例
```

| 項目 | `random` | `retrieved` |
|---|---|---|
| ディストラクタ | 無作為 | 検索の上位。`--mix-random`（既定 0.2）の確率で各枠を無作為の文書に置き換える |
| ゴールデン抜きの例 | `--oracle-prob` の抽選 | 検索が正解の文書を上位 (k+1) 件に入れなかった質問 |
| 答えを含む別の文書 | 確認しない | 本文が正解と同じ・回答の引用を含む文書は文脈から外し、次の順位を繰り上げる |
| 文書の並び | 入れ替え | 入れ替え（`--doc-order rank` で検索の順位のまま） |
| 双子の差し替え候補 | 質問に字面が近い文書 | 検索で次点だった文書 |

```bash
# 埋め込み・リランカーの接続先は .env で指定する（.env.example の例は ruri-v3-310m / ruri-v3-reranker-310m）
python scripts/build_raft.py data/input/corpus.jsonl -o data/output/sft_hardneg.jsonl \
  --max-doc-chars 450 --distractor-mode retrieved --question-check on \
  --absent-mode refuse --twin-ratio 0.25 --context-format bullets \
  --user-instruction rag_refuse --val-split 0.1 --keep-intermediate
```

検索の結果は `data/.checkpoint/{stem}_retrieved.jsonl` に 1 問 1 行で残り、`--resume` で使い回す。
検索の設定（方式・件数・モデル・コーパス）が変わると自動で引き直す。コーパスの埋め込みは `data/.cache/` に置く。
`--dry-run` はネットワークを使わず BM25 だけで順位を決める。

### 設計上の注意

- **ゴールデン抜きの例は自然にはほとんど集まらない。** 合成した質問は本文に字面が近く、検索が正解の文書を
  上位に入れやすい。拒否例は双子（`--twin-ratio`）で作る。
- **検索の順位のまま並べると正解が 1 番目に偏る。** リランカーが正解の文書をほぼ 1 位に置くため。
  位置の手掛かりを残さないよう、既定は入れ替えにしてある。
- **答えを含む別のチャンクを外す処理は必須である。** 同じ文書から切り出したチャンクは内容が重なることがあり、
  検索の上位には回答の引用をすべて含む別のチャンクが入りやすい。無作為抽出ではほとんど当たらない。

## 質問の判定（`--question-check on`）

Phase1 の検証は「空でないこと」だけである。`on` にすると、回答ができた質問を 1 問ずつ判定し、
`{stem}_question_review.jsonl` に理由つきで書く。文字列の照合と検索の結果だけで決まる（LLM は使わない）。

| 判定 | 条件 |
|---|---|
| reject 自動除外 | 指示語で文書を指す（「文書内の」「文書に記載されている」「提示された」）/ 答えが質問文に入っている / ほかの質問と同じ文面 |
| review 要確認 | 文脈に頼る語（「前項」「本研究」）/ 本文と 30 文字以上一致 / ほぼ同じ文面 / 正解の文書が検索の候補に入らない / 長さが範囲外 |
| accept 自動採用 | 上のどれにも当たらない |

- 要確認は既定で学習例に使わない（`--review-policy keep` で使う）。
- 人が確認した結果は、確認用ファイルの `verdict` を書き換えて `--question-overrides` に渡すと反映される。
- 「検索の上位に同じ内容の文書がある」は注記として理由に残すだけで、判定は変えない（質問の不備ではないため）。
- 見ているのは形だけである。質問が自然か、回答が正しいかは判定していない。

## 入力 / 出力

- **入力**: チャンク済み JSONL（`cpt-data-pipeline` の `_preprocessed.jsonl` 互換）
  ```json
  {"text": "チャンク本文", "id": "doc_a", "meta": {"source_file": "...", "section_path": "..."}}
  ```
- **出力**: トレーナー契約（`継続事前学習/sft.yaml` / `build_sft_dataset`）に一致
  ```json
  {"messages": [{"role": "user", "content": "<文脈+質問>"}, {"role": "assistant", "content": "<CoT回答>"}]}
  ```
  chat template はトレーナー側で適用される（本パイプラインでは事前適用しない）。

## セットアップ

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env   # vLLM エンドポイント等を設定
```

`.env` の主な項目:
- `LLM_PROVIDER=vllm`（既定）/ `VLLM_ENDPOINT` / `VLLM_MODEL`
- `SWALLOW_TOKENIZER`（長さフィルタ用・学習側と一致させる）
- オフライン実行時は `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`（トークナイザを事前キャッシュ）

## 実行

```powershell
# 本番（vLLM 起動済み）。CPTの大きいチャンク(~1300字)は --max-doc-chars で割り直す（下記「チャンクサイズ」参照）
python scripts/build_raft.py data/input/corpus.jsonl -o data/output/sft_dataset.jsonl `
    --max-doc-chars 450 --questions-per-chunk 3 --num-distractors 4 --oracle-prob 0.8 `
    --max-len 2048 --concurrency 8 --keep-intermediate

# vLLM 不要のスモーク（決定的モック・HF トークナイザも不要）
python scripts/build_raft.py tests/fixtures/tiny_corpus.jsonl -o data/output/smoke.jsonl `
    --dry-run --limit 4 --val-split 0.25

# 生成コスト/所要時間の見積
python scripts/raft/estimate.py data/input/corpus.jsonl --questions-per-chunk 3 --concurrency 8
```

主なオプション: `--distractor-mode` `--question-check` `--oracle-prob`(P) `--num-distractors`(k) `--questions-per-chunk` `--seed`
`--max-len` `--oversize-policy {reduce,drop}` `--min-distractors` `--val-split` `--resume`
`--max-doc-chars`(再分割) `--max-goldens`(件数上限) `--dry-run` `--keep-intermediate`。

### system プロンプト（学習データへの注入）

`--system-prompt` で出力 `messages` の先頭に system を付与できる。プリセット名（`prompts.py` の
`SFT_SYSTEM_PROMPTS`）か任意の文字列を指定:

- `raft` … RAG/RAFT契約（文書を根拠に逐語引用＋CoTで回答、無ければ知識で）
- `rag_strict` … 文書のみで回答（外部知識を使わない）
- `assistant_ja` … Swallowテンプレ既定と同等の汎用ペルソナ

`--system-dropout 0.15` で一定割合を system 無しで出力し、推論時の system ブレに頑健化（決定的）。

> **重要（train/serve 一致）**: 学習データに system が無いと、Swallow の chat template が既定の
> 「あなたは誠実で優秀な日本人のアシスタントです。」を自動注入する。推論時に独自 system を使うなら、
> **同じ system を `--system-prompt` で学習データにも入れる**こと。

## 長さフィルタ（重要）

`--max-len`（既定 2048）超過の例は、学習側 **Swallow トークナイザ**で `apply_chat_template` 後に計測し処理する:
- `reduce`（既定）: ディストラクタを末尾から減らして収める（ゴールデンは最後まで残す）
- `drop`: 破棄

生成用 `VLLM_MODEL` ではなく学習用トークナイザで測るため、トレーナーの切り捨て境界と一致する。

## チャンクサイズ（重要・運用知見）

RAFT は1つの user メッセージに「ゴールデン + 複数ディストラクタ」を並べるため、**1チャンクが大きいと
`--max-len`(2048) に収まらない**。実測（Swallowトークナイザ）:

- CPTの `corpus_clean.jsonl` は1チャンク ≈ **960トークン**（~1300字）
- ゴールデン単体 + CoT ≈ 1281トークン（収まる）／ **ゴールデン + ディストラクタ1件 ≈ 2282 → 超過**

→ そのままだと oversize でほぼ全滅する。**`--max-doc-chars 450` で読み込み時に再分割**すると、
ゴールデン + ディストラクタ4件 + CoT が 2048 以内に収まる（実測 37/37 採用・最大2030トークン・dropped 0）。
より余裕が欲しい場合は 400、k(`--num-distractors`)を 3 に下げるのも有効。

## チェックポイント / 再開

Phase1/2 は **1レコードずつ追記**（`data/.checkpoint/`）。途中でvLLM再起動やCtrl-Cが起きても完了分は残り、
`--resume` で未処理分のみ再生成する（H1）。本番は `--keep-intermediate` を付けて中間成果を保持すること。
起動時に vLLM 疎通とモデル一致を **preflight** で確認し、エンドポイント誤り等は生成前に中断する。

## テスト

```powershell
pytest tests/ -q
```

`tests/` は HF/vLLM 不要（pydantic のみ）。整形・合成の決定性・出力契約・長さゲート・拒否例の検証・検索の順位・質問の判定を検証する（LLM と検索の接続先は呼ばない）。

## ディレクトリ

```
scripts/
  llm_factory.py   # cpt からコピー（プロバイダ抽象）
  prompts.py       # RAFT 用プロンプト
  build_raft.py    # オーケストレータ CLI
  raft/            # schemas, corpus, question_gen, answer_gen, assemble, export, tokenizer_utils, checkpoint, formatting, estimate,
                   # refusal（拒否例）, retrieval（検索）, question_check（質問の判定）
tests/             # fixtures + 単体/契約テスト
data/{input,output,.checkpoint}/
```
