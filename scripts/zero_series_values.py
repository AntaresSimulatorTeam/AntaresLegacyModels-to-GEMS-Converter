import argparse
from pathlib import Path
import re

def replace_numbers_with_zero(text: str) -> str:
    number_pattern = re.compile(r"(?<![\w.-])[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?(?![\w.-])")
    return number_pattern.sub("0", text)


def process_file(path: Path, backup: bool = True) -> int:
    content = path.read_text(encoding="utf-8")
    new_content = replace_numbers_with_zero(content)
    if new_content != content:
        if backup:
            backup_path = path.with_suffix(path.suffix + ".bak")
            path.rename(backup_path)
            path.write_text(new_content, encoding="utf-8")
        else:
            path.write_text(new_content, encoding="utf-8")
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replace all numeric values with 0 in every series.txt file under a directory."
    )
    parser.add_argument(
        "base_dir",
        nargs="?",
        default=r"C:\\Users\\jeannecor\\Documents\\1-PROJECTS\\OPEN SOURCE\\antares-studies-converted\\BP25\\input\\thermal\\series\\z_effacement\\",
        help="Root directory to search for series.txt files.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not keep a .bak backup of modified files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show files that would be changed without writing.",
    )
    args = parser.parse_args()

    root = Path(args.base_dir)
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Le dossier spécifié n'existe pas ou n'est pas un répertoire: {root}")

    files = sorted(root.rglob("series.txt"))
    if not files:
        print(f"Aucun fichier series.txt trouvé sous {root}")
        return

    modified = 0
    for file_path in files:
        if args.dry_run:
            print(f"[dry-run] {file_path}")
            continue

        changed = process_file(file_path, backup=not args.no_backup)
        if changed:
            print(f"Modifié: {file_path}")
            modified += 1
        else:
            print(f"Pas de changement: {file_path}")

    print(f"Terminé: {modified} fichier(s) modifié(s)")


if __name__ == "__main__":
    main()
