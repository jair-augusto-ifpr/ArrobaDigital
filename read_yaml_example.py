"""Valida config.yaml (existência + parse YAML). Usado por `make test`."""
import sys
from pathlib import Path

import yaml


def main() -> None:
    root = Path(__file__).resolve().parent
    path = root / "config.yaml"
    if not path.is_file():
        print(f"ERRO: {path} não encontrado", file=sys.stderr)
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        yaml.safe_load(f)
    print("config.yaml OK")


if __name__ == "__main__":
    main()
