import urllib.request
import json
import time
import sys

def pull_model(name):
    print(f"Requesting pull for {name}...")
    req = urllib.request.Request(
        "http://localhost:11434/api/pull",
        data=json.dumps({"name": name, "stream": True}).encode(),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        for line in resp:
            if line:
                chunk = json.loads(line.decode())
                status = chunk.get("status", "")
                completed = chunk.get("completed", 0)
                total = chunk.get("total", 0)
                if total > 0:
                    pct = (completed / total) * 100
                    mb_c = completed / (1024**2)
                    mb_t = total / (1024**2)
                    sys.stdout.write(f"\r{status}: {pct:.1f}% ({mb_c:.1f}/{mb_t:.1f} MB)")
                    sys.stdout.flush()
                else:
                    sys.stdout.write(f"\r{status}")
                    sys.stdout.flush()
    print("\nPull complete!")

if __name__ == "__main__":
    pull_model("qwen2.5:7b")
