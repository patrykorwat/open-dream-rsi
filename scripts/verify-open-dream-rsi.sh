#!/usr/bin/env bash
# verify-open-dream-rsi.sh — run INSIDE the hermes container (docker exec -i hermes bash /tmp/verify-odr.sh)
set -e
echo "PYTHONPATH=$PYTHONPATH"
python3 -c 'import open_dream_rsi; print("import OK:", open_dream_rsi.__file__)'
python3 - <<'PYEOF'
import subprocess, json
req = b"\n".join([
    json.dumps({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"verify","version":"1"}}}).encode(),
    json.dumps({"jsonrpc":"2.0","method":"notifications/initialized"}).encode(),
    json.dumps({"jsonrpc":"2.0","id":2,"method":"tools/list"}).encode(),
])
r = subprocess.run(["python3","-m","open_dream_rsi","mcp"], input=req,
                   capture_output=True, timeout=30)
assert r.returncode == 0, r.stderr.decode()[:300]
for line in r.stdout.decode().splitlines():
    d = json.loads(line)
    if d.get("id") == 2:
        names = [t["name"] for t in d["result"]["tools"]]
        assert "odr_run_once" in names, names
        print("MCP handshake OK, tools:", names)
PYEOF
hermes mcp list 2>&1 | tail -3
echo "ALL GOOD"
