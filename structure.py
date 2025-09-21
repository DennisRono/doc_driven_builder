#!/usr/bin/env python3

import os
from pathlib import Path
from datetime import datetime

# --- Configuration ---
EXCLUDED_DIRS = {
    "__pycache__",
    ".git",
    ".svn",
    ".hg",
    ".idea",
    ".vscode",
    ".venv",
    "env",
    "venv",
    "node_modules",
    "dist",
    "build",
}

EXCLUDED_FILES = {"Thumbs.db", ".DS_Store"}

README_NAME = "STRUCTURE.md"
TITLE = "# Project Structure\n"


def walk_directory(root: Path):
    """Yield all files and dirs under root (depth-first), skipping excluded ones."""
    for dirpath, dirnames, filenames in os.walk(root):
        # filter out excluded dirs in-place so os.walk won’t descend into them
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]

        rel_dir = Path(dirpath).relative_to(root)

        for f in sorted(filenames):
            if f in EXCLUDED_FILES or f == README_NAME:
                continue
            yield rel_dir / f


def build_tree(root: Path) -> str:
    """Return a pretty ascii tree of root, skipping junk."""
    lines = []

    def walk(path: Path, prefix=""):
        items = [p for p in sorted(path.iterdir()) if p.name not in EXCLUDED_FILES]
        items = [p for p in items if p.name not in EXCLUDED_DIRS]
        for idx, item in enumerate(items):
            connector = "└── " if idx == len(items) - 1 else "├── "
            lines.append(f"{prefix}{connector}{item.name}")
            if item.is_dir():
                extension = "    " if idx == len(items) - 1 else "│   "
                walk(item, prefix + extension)

    lines.append(root.name)  # show the project folder itself
    walk(root)
    return "\n".join(lines)


def generate_readme():
    root = Path(__file__).parent.resolve()
    print(root)
    tree = build_tree(root)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    content = (
        f"{TITLE}\n"
        f"_Generated on {timestamp}_\n\n"
        "```\n"
        f"{tree}\n"
        "```\n"
    )

    with open(root / README_NAME, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"README.md generated at {root}")


if __name__ == "__main__":
    generate_readme()
