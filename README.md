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

主なオプション: `--oracle-prob`(P) `--num-distractors`(k) `--questions-per-chunk` `--seed`
`--max-len` `--oversize-policy {reduce,drop}` `--min-distractors` `--val-split` `--resume`
`--system-prompt`（literal か `SYSTEM_PROMPTS` のキー）`--dry-run` `--keep-intermediate`。

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

`tests/` は HF/vLLM 不要（pydantic のみ）。整形・合成の決定性・出力契約・長さゲートを検証する。

## ディレクトリ

```
scripts/
  llm_factory.py   # cpt からコピー（プロバイダ抽象）
  prompts.py       # RAFT 用プロンプト
  build_raft.py    # オーケストレータ CLI
  raft/            # schemas, corpus, question_gen, answer_gen, assemble, export, tokenizer_utils, checkpoint, formatting, estimate
tests/             # fixtures + 単体/契約テスト
data/{input,output,.checkpoint}/
```
