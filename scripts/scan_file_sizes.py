"""
Scan all files in the project to check GitHub file size thresholds:
- > 100 MB: Hard reject on GitHub (blocks git push)
- > 50 MB: Warning on GitHub
- > 25 MB: Substantial size
"""

import os
from pathlib import Path

files_over_100m = []
files_over_50m = []
files_over_25m = []

# Exclude .venv and training datasets that are in .gitignore
ignored_roots = ['.venv', '.git', '__pycache__', 'CMAPSSData']

for root, dirs, files in os.walk('.'):
    if any(x in root for x in ignored_roots):
        continue
    for f in files:
        p = Path(root) / f
        try:
            sz = p.stat().st_size
            sz_mb = sz / (1024 * 1024)
            if sz_mb > 100:
                files_over_100m.append((str(p), sz_mb))
            elif sz_mb > 50:
                files_over_50m.append((str(p), sz_mb))
            elif sz_mb > 25:
                files_over_25m.append((str(p), sz_mb))
        except Exception:
            pass

print("=" * 65)
print("GITHUB FILE SIZE AUDIT")
print("=" * 65)
print(f"Files > 100 MB (GitHub Hard Block): {len(files_over_100m)}")
for p, sz in files_over_100m:
    print(f"  [BLOCKER] {p}: {sz:.2f} MB")

print(f"\nFiles 50 - 100 MB (GitHub Warning): {len(files_over_50m)}")
for p, sz in files_over_50m:
    print(f"  [WARNING] {p}: {sz:.2f} MB")

print(f"\nFiles 25 - 50 MB (Large Files): {len(files_over_25m)}")
for p, sz in files_over_25m:
    print(f"  [NOTICE]  {p}: {sz:.2f} MB")
print("=" * 65)
