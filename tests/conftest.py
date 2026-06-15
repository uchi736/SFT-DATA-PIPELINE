import sys
from pathlib import Path

# scripts/ を import パスに追加（raft パッケージ / prompts / llm_factory を解決）
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
