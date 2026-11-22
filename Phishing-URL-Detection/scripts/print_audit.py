import json
from pathlib import Path

from src.preprocessing import load_dataset


DATA = "data/raw/PhiUSIIL_Phishing_URL_Dataset.csv"
OUT = Path("reports/dataset_audit.json")


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {OUT}")

    data = load_dataset("data/raw/PhiUSIIL_Phishing_URL_Dataset.csv")
    audit = data.attrs["audit"]
    text = json.dumps(audit, indent=2, ensure_ascii=False)
    OUT.write_text(text, encoding="utf-8")
    print(text)
    print("Saved:", OUT.resolve())


if __name__ == "__main__":
    main()
