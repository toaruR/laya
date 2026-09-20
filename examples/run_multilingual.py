"""Run the standalone Laya multilingual checkpoint on Japanese text."""

import argparse
import json
from pathlib import Path
import sys

import laya


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        default="3月分の請求が二重になっています。至急、重複分を返金してください。",
        help="Text to classify.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    model_dir = repo_root / "models" / "laya-multilingual"
    agent = laya.load(str(model_dir))
    questions = {
        "department": {
            "type": "choice",
            "instructions": "この問い合わせを担当すべき部門はどこですか？",
            "criteria": {
                "billing": "請求、支払い、返金",
                "technical": "不具合、障害、システムエラー",
                "sales": "価格、新規契約",
                "other": "その他",
            },
        },
        "urgency": {
            "type": "score",
            "instructions": "この問い合わせの緊急度はどの程度ですか？",
            "criteria": ["緊急ではない", "早めの対応が必要", "重大または対応を妨げる問題"],
        },
        "refund_requested": {
            "type": "noul",
            "instructions": "利用者は明示的に返金を求めていますか？",
        },
    }

    result = agent.predict({"body": args.text}, questions)
    print(f"device: {agent.device}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
