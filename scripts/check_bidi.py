#!/usr/bin/env python3
"""Fail if bidi control characters are present in tracked files.

Checks for Unicode bidi control characters that can obscure code review.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys
from typing import Iterable

BIDI_CODEPOINTS = {
    0x202A,  # LRE
    0x202B,  # RLE
    0x202D,  # LRO
    0x202E,  # RLO
    0x202C,  # PDF
    0x2066,  # LRI
    0x2067,  # RLI
    0x2068,  # FSI
    0x2069,  # PDI
}


def iter_files(args: list[str]) -> Iterable[pathlib.Path]:
    if args:
        for name in args:
            yield pathlib.Path(name)
        return

    try:
        result = subprocess.run(
            ["git", "ls-files"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(f"Failed to list tracked files: {exc.stderr}\n")
        sys.exit(2)

    for line in result.stdout.splitlines():
        if line:
            yield pathlib.Path(line)


def has_bidi(text: str) -> bool:
    return any(ord(ch) in BIDI_CODEPOINTS for ch in text)


def main() -> int:
    offenders: list[str] = []
    for path in iter_files(sys.argv[1:]):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        if has_bidi(text):
            offenders.append(str(path))

    if offenders:
        sys.stderr.write("Bidi control characters detected in:\n")
        for name in offenders:
            sys.stderr.write(f"  - {name}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
