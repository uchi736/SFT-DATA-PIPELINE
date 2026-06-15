"""
フェーズ別チェックポイント（cpt-data-pipeline/batch_pipeline.py の方式を踏襲）。

- 各フェーズの成果を {stem}_{phase}.jsonl に完全書き込み
- {stem}_state.json の completed_phases に名前を追記
- --resume 時は完了済みフェーズをスキップし、その成果 JSONL を読み戻す

cpt の _load_checkpoint は「最新フェーズのみ」復元だが、RAFT は特定フェーズを
個別に読み戻したいので load_phase(phase) を提供する。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional


def phase_path(checkpoint_dir: Path, stem: str, phase: str) -> Path:
    return checkpoint_dir / f"{stem}_{phase}.jsonl"


def _state_path(checkpoint_dir: Path, stem: str) -> Path:
    return checkpoint_dir / f"{stem}_state.json"


def read_state(checkpoint_dir: Path, stem: str) -> dict:
    sp = _state_path(checkpoint_dir, stem)
    if not sp.exists():
        return {}
    return json.loads(sp.read_text(encoding="utf-8"))


def completed_phases(checkpoint_dir: Path, stem: str) -> list[str]:
    return read_state(checkpoint_dir, stem).get("completed_phases", [])


def is_completed(checkpoint_dir: Path, stem: str, phase: str) -> bool:
    return phase in completed_phases(checkpoint_dir, stem)


def save_phase(checkpoint_dir: Path, stem: str, phase: str, records: list[dict]) -> None:
    """フェーズ成果を JSONL に書き出し、state を更新。"""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    data_path = phase_path(checkpoint_dir, stem, phase)
    with open(data_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    sp = _state_path(checkpoint_dir, stem)
    state = read_state(checkpoint_dir, stem)
    completed = state.get("completed_phases", [])
    if phase not in completed:
        completed.append(phase)
    state["completed_phases"] = completed
    state[f"{phase}_count"] = len(records)
    sp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  checkpoint: {phase} ({len(records)}件)")
    sys.stdout.flush()


def load_phase(checkpoint_dir: Path, stem: str, phase: str) -> Optional[list[dict]]:
    """指定フェーズ成果 JSONL を読み戻す。無ければ None。"""
    data_path = phase_path(checkpoint_dir, stem, phase)
    if not data_path.exists():
        return None
    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    print(f"  checkpoint復元: {phase} ({len(records)}件)")
    sys.stdout.flush()
    return records


def read_phase(checkpoint_dir: Path, stem: str, phase: str) -> list[dict]:
    """フェーズ JSONL を静かに読む（未完了でも部分結果を返す）。無ければ []。"""
    data_path = phase_path(checkpoint_dir, stem, phase)
    if not data_path.exists():
        return []
    out: list[dict] = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def existing_keys(checkpoint_dir: Path, stem: str, phase: str, key: str) -> set:
    """フェーズ JSONL に既に書かれたレコードの指定キー集合（resume のスキップ用）。"""
    return {r[key] for r in read_phase(checkpoint_dir, stem, phase) if key in r}


def mark_completed(checkpoint_dir: Path, stem: str, phase: str, count: int) -> None:
    """フェーズを完了としてマーク（逐次追記後に全件揃ったら呼ぶ）。"""
    sp = _state_path(checkpoint_dir, stem)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state = read_state(checkpoint_dir, stem)
    completed = state.get("completed_phases", [])
    if phase not in completed:
        completed.append(phase)
    state["completed_phases"] = completed
    state[f"{phase}_count"] = count
    sp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  checkpoint: {phase} 完了 ({count}件)")
    sys.stdout.flush()


class Appender:
    """フェーズ JSONL へ1レコードずつ追記する逐次チェックポイント。

    resume=False では既存ファイルを truncate、resume=True では追記継続。
    各 write で flush するので、途中クラッシュでも完了分は残る（H1 対策）。
    """

    def __init__(self, checkpoint_dir: Path, stem: str, phase: str, resume: bool):
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.path = phase_path(checkpoint_dir, stem, phase)
        if not resume and self.path.exists():
            self.path.unlink()
        self._f = open(self.path, "a", encoding="utf-8")

    def write(self, rec: dict) -> None:
        self._f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass


def reset(checkpoint_dir: Path, stem: str) -> None:
    """state と全フェーズ JSONL を削除（--resume 無し実行用）。"""
    sp = _state_path(checkpoint_dir, stem)
    phases = completed_phases(checkpoint_dir, stem)
    for ph in phases:
        p = phase_path(checkpoint_dir, stem, ph)
        if p.exists():
            p.unlink()
    if sp.exists():
        sp.unlink()
