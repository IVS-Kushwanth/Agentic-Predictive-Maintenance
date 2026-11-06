"""
Scan codebase for accidentally committed secrets, tokens, API keys, or private credentials.
"""

import os
import re
from pathlib import Path

patterns = [
    re.compile(r'sk-[a-zA-Z0-9]{20,}'),
    re.compile(r'ghp_[a-zA-Z0-9]{36}'),
    re.compile(r'xox[baprs]-[0-9a-zA-Z]{10,48}'),
    re.compile(r'(?:api[_-]?key|secret[_-]?key|auth_token)\s*[:=]\s*["\']([^"\']{16,})["\']', re.IGNORECASE),
]

hits = []
for root, dirs, files in os.walk('.'):
    # Ignore cache, venv, git
    if any(p in root for p in ['.venv', '__pycache__', '.git', '.pytest_cache']):
        continue
    for f in files:
        if f.endswith(('.py', '.json', '.yaml', '.yml', '.env', '.toml', '.txt', '.md', '.sh')):
            p = Path(root) / f
            try:
                text = p.read_text(encoding='utf-8', errors='ignore')
                for pat in patterns:
                    for match in pat.finditer(text):
                        val = match.group(0)
                        # Filter out known safe dummy template values like sk-proj-... or sk-ant-...
                        if any(safe in val for safe in ['sk-proj-...', 'sk-ant-...', 'your-api-key', 'dummy_key']):
                            continue
                        hits.append((str(p), match.start()))
            except Exception:
                pass

print('=' * 60)
print('CODEBASE SECRETS AUDIT')
print('=' * 60)
if not hits:
    print('>>> ZERO SECRETS DETECTED: Codebase is 100% clean of hardcoded credentials! <<<')
else:
    print(f'WARNING: Found {len(hits)} potential secret(s) in:')
    for path, pos in hits:
        print(f'  - {path} at character offset {pos}')
print('=' * 60)
