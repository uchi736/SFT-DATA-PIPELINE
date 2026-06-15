"""
LLM 生成フェーズ共通ユーティリティ（非同期実行・修復リトライ・vLLM guided_json）。

pydantic-ai の output_type による構造化出力を主経路としつつ、以下を上乗せする:
  - タイムアウト時に1回再試行
  - 出力検証(validate)失敗時に修復ヒントを付けて再試行
  - vLLM 限定で guided_json（json-schema guided decoding）を任意有効化（RAFT_GUIDED_JSON=1）

guided_json は pydantic-ai のバージョン差で model_settings のキーが異なりうるため、
渡して失敗したら settings 無しで自動フォールバックし、既定経路を壊さない。
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Awaitable, Callable, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))  # scripts/
from llm_factory import get_provider  # noqa: E402


def guided_settings(schema_model) -> Optional[dict]:
    """vLLM 用 guided_json の model_settings を返す（opt-in / 非vLLMでは None）。"""
    if get_provider() != "vllm" or os.getenv("RAFT_GUIDED_JSON", "0") != "1":
        return None
    try:
        schema = schema_model.model_json_schema()
    except Exception:
        return None
    return {"extra_body": {"guided_json": schema}}


async def run_structured(
    agent,
    prompt: str,
    timeout: float,
    model_settings: Optional[dict] = None,
    validate: Optional[Callable[[object], bool]] = None,
    repair_hint: str = "",
    max_attempts: int = 2,
):
    """構造化出力を取得。失敗（timeout/parse/検証）時は修復ヒントを付けて再試行。

    成功で output オブジェクト、最終失敗で None を返す（例外は呑んでバッチを止めない）。
    """
    cur_prompt = prompt
    for attempt in range(1, max_attempts + 1):
        try:
            coro = _agent_run(agent, cur_prompt, model_settings)
            result = await asyncio.wait_for(coro, timeout=timeout)
            output = result.output
            if validate is None or validate(output):
                return output
            # 検証失敗 → 修復ヒントを付けて再試行
            cur_prompt = f"{prompt}\n\n{repair_hint}".strip()
        except asyncio.TimeoutError:
            print(f"  [warn] LLM timeout (attempt {attempt})", file=sys.stderr)
            cur_prompt = prompt  # タイムアウトは同一プロンプトで再試行
        except Exception as e:  # JSON/スキーマ違反含む
            print(f"  [warn] LLM error (attempt {attempt}): {e}", file=sys.stderr)
            cur_prompt = f"{prompt}\n\n{repair_hint}".strip()
    return None


async def _agent_run(agent, prompt: str, model_settings: Optional[dict]):
    """model_settings 付きで実行。未対応/プロバイダ拒否時は settings 無しに自動フォールバック。

    pydantic-ai が古く model_settings 自体を受け付けない場合は TypeError、
    vLLM が guided_json を拒否する場合は 400(BadRequest) 等が来るため、双方とも
    settings を外して素のパスへ退避する（既定経路を壊さない / 無限リトライを防ぐ）。
    """
    if model_settings:
        try:
            return await agent.run(prompt, model_settings=model_settings)
        except TypeError:
            pass  # 古い/異なる pydantic-ai → settings 無しで実行
        except Exception as e:  # 例: vLLM が guided_json を 400 で拒否
            print(f"  [warn] model_settings 拒否 ({type(e).__name__}) → 無効化して再試行",
                  file=sys.stderr)
    return await agent.run(prompt)


async def run_incremental(units, make_coro, concurrency: int, on_done) -> None:
    """各ユニットを並行実行し、完了するたびに on_done(unit, result) を呼ぶ。

    on_done は逐次チェックポイントの書き込み点（H1）。例外は内部で握りつぶし
    result=None として on_done に渡すため、単一ユニットの失敗がバッチ全体を止めない（H2）。
    on_done は同期関数（単一スレッド asyncio 上で順次呼ばれるのでファイル追記に安全）。
    """
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _one(u):
        async with sem:
            try:
                r = await make_coro(u)
            except Exception as e:
                print(f"  [warn] ユニット処理エラー: {e}", file=sys.stderr)
                r = None
            on_done(u, r)

    await asyncio.gather(*[_one(u) for u in units], return_exceptions=True)


async def gather_bounded(coros: list[Awaitable], concurrency: int) -> list:
    """セマフォで並行数を絞って gather（return_exceptions で1件の失敗が全体を止めない）。"""
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _wrapped(c):
        async with sem:
            return await c

    return await asyncio.gather(*[_wrapped(c) for c in coros], return_exceptions=True)
