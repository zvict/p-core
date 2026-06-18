#!/usr/bin/env python3
"""Static-site sanity check for the P-CORE project page.

1. Every local asset referenced by index.html exists on disk.
2. No leftover template placeholder tokens remain in index.html.
3. No file > 100 MB (GitHub limit); web videos < 20 MB.

Exit 0 = pass, 1 = fail.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
INDEX = os.path.join(ROOT, "index.html")

RESERVED_MISSING = set()  # the paper PDF is present

PLACEHOLDER_TOKENS = [
    "PAPER_TITLE", "AUTHOR_NAMES", "BRIEF_DESCRIPTION", "YOUR_DOMAIN",
    "KEYWORD1", "FIRST_AUTHOR", "SECOND_AUTHOR", "CONFERENCE_NAME",
    "INSTITUTION_OR_LAB_NAME", "RESEARCH_AREA", "Lorem ipsum",
    "TODO: author homepage", "favicon.ico", "Statcounter",
]


def main():
    with open(INDEX, encoding="utf-8") as f:
        html = f.read()

    failures = []

    for ref in re.findall(r'(?:src|href)="([^"]+)"', html):
        if ref.startswith(("http://", "https://", "#", "mailto:", "data:")):
            continue
        if ref in RESERVED_MISSING:
            continue
        if not os.path.exists(os.path.join(ROOT, ref)):
            failures.append(f"Missing asset: {ref}")

    for tok in PLACEHOLDER_TOKENS:
        if tok in html:
            failures.append(f"Leftover placeholder token: {tok!r}")

    for dirpath, _dirs, names in os.walk(ROOT):
        if ".git" in dirpath.split(os.sep):
            continue
        for name in names:
            fp = os.path.join(dirpath, name)
            size = os.path.getsize(fp)
            rel = os.path.relpath(fp, ROOT)
            if size > 100 * 1024 * 1024:
                failures.append(f"File over 100MB GitHub limit: {rel} ({size // 1048576}MB)")
            elif rel.startswith("static/videos" + os.sep) and size > 20 * 1024 * 1024:
                failures.append(f"Web video over 20MB: {rel} ({size // 1048576}MB)")

    if failures:
        print("FAIL")
        for fl in failures:
            print("  -", fl)
        return 1
    print("PASS - all asset references resolve, no placeholders, sizes OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
