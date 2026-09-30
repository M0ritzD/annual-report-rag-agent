#!/usr/bin/env bash
# Lädt die Demo-Geschäftsberichte (öffentliche PDFs, ca. 40 MB) nach data/reports/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/reports
python3 - <<'EOF'
import json, urllib.request, pathlib
for r in json.load(open("data/reports.json")):
    target = pathlib.Path("data/reports") / r["file"]
    if target.exists():
        print(f"✔ {r['file']} (vorhanden)"); continue
    print(f"↓ {r['company']}: {r['url']}")
    req = urllib.request.Request(r["url"], headers={"User-Agent": "Mozilla/5.0"})
    target.write_bytes(urllib.request.urlopen(req, timeout=300).read())
EOF
