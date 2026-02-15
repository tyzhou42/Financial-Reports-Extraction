import os
from pathlib import Path
import logging
from typing import List

from logging_utils import setup_logging


logger = logging.getLogger(__name__)

def generate_tree(root: Path, prefix: str = "") -> str:
    """生成目录树字符串"""
    entries = sorted(root.iterdir())
    tree_str = ""
    for idx, entry in enumerate(entries):
        connector = "└── " if idx == len(entries) - 1 else "├── "
        tree_str += f"{prefix}{connector}{entry.name}\n"
        if entry.is_dir():
            extension = "    " if idx == len(entries) - 1 else "│   "
            tree_str += generate_tree(entry, prefix + extension)
    return tree_str

def collect_py_files(root: Path) -> List[Path]:
    """收集所有 .py 文件路径"""
    return [p for p in root.rglob("*.py") if p.is_file()]


def generate_markdown(root: Path, output_file: str = "project_dump.md") -> None:
    py_files = collect_py_files(root)

    with open(output_file, "w", encoding="utf-8") as out:
        # 项目目录树
        out.write("# 项目目录结构\n\n")
        out.write("```\n")
        out.write(generate_tree(root))
        out.write("```\n\n")

        # 文件内容
        out.write("# Python 文件内容\n\n")
        for file in py_files:
            rel_path = file.relative_to(root)
            out.write(f"## {rel_path}\n\n")
            out.write("```python\n")
            try:
                content = file.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = file.read_text(encoding="latin-1")
            out.write(content)
            out.write("\n```\n\n")

    logger.info("Generated project dump: %s", output_file)


if __name__ == "__main__":
    setup_logging(log_file=Path(__file__).resolve().parent / "logs" / "project_exporter.log")
    current_dir = Path(__file__).parent.resolve()
    generate_markdown(current_dir, "project_dump.md")
