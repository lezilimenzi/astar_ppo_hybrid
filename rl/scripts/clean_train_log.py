#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clean a `script`-captured training log: strip ANSI codes and rich progress bars,
keep only real log content (tables, eval lines, episode summaries)."""
import re
import sys

ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")
BAR_CHAR = "━"  # ━ heavy horizontal (rich progress bar)


def clean_lines(raw: str) -> list[str]:
    segs = re.split(r"[\r\n]", raw)  # progress bars overwrite with \r
    out: list[str] = []
    for s in segs:
        s = ANSI.sub("", s).rstrip()
        if not s.strip():
            continue
        if s.startswith("Script started") or s.startswith("Script done"):
            continue
        if BAR_CHAR in s or "it/s" in s:
            continue
        out.append(s)
    return out


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    with open(src, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()
    out = clean_lines(raw)
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out) + "\n")
    print(f"{src}: cleaned {len(out)} lines -> {dst}")


if __name__ == "__main__":
    main()
