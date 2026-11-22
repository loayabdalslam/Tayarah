import os
import sys
import json
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from fastapi.testclient import TestClient
from tayarah.server.app import app

client = TestClient(app)

def test_endpoints():
    print("==================================================")
    print("🧪 Running Comprehensive Tayarah Server Tests...")
    print("==================================================")

    # 1. Root & Health
    resp = client.get("/")
    assert resp.status_code == 200, f"Root failed: {resp.status_code}"
    assert "TAYARAH" in resp.text, "Root HTML missing branding"
    print("✅ GET / (Dashboard HTML): OK")

    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    print(f"✅ GET /health: OK ({data['runtime']})")

    resp = client.get("/api/telemetry")
    assert resp.status_code == 200
    print(f"✅ GET /api/telemetry: OK ({resp.json()['status']})")

    # 2. Ollama API
    resp = client.get("/api/version")
    assert resp.status_code == 200
    print(f"✅ GET /api/version: OK ({resp.json()['version']})")

    resp = client.get("/api/tags")
    assert resp.status_code == 200
    tags = resp.json().get("models", [])
    assert len(tags) >= 1
    print(f"✅ GET /api/tags: OK ({len(tags)} models listed)")

    # 3. OpenAI / vLLM Models
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    models = resp.json().get("data", [])
    assert len(models) >= 1
    print(f"✅ GET /v1/models: OK ({len(models)} models listed)")

    # 4. Ollama /api/generate Streaming
    print("\n--- Testing Ollama Streaming (/api/generate) ---")
    gen_payload = {
        "model": "tayarah",
        "prompt": "Say hello in 3 words:",
        "stream": True,
        "options": {"num_predict": 10}
    }
    with client.stream("POST", "/api/generate", json=gen_payload) as stream_resp:
        assert stream_resp.status_code == 200
        chunks = []
        for line in stream_resp.iter_lines():
            if line:
                pkt = json.loads(line)
                if pkt.get("response"):
                    chunks.append(pkt["response"])
                if pkt.get("done"):
                    print(f"   Done packet received: eval_count={pkt.get('eval_count')}")
        full_text = "".join(chunks)
        print(f"✅ Ollama Streamed Output: '{full_text.strip()}'")

    # 5. vLLM / OpenAI /v1/chat/completions Streaming
    print("\n--- Testing vLLM / OpenAI SSE Streaming (/v1/chat/completions) ---")
    chat_payload = {
        "model": "tayarah",
        "messages": [{"role": "user", "content": "Say 'Mach speed'"}],
        "stream": True,
        "max_tokens": 10
    }
    with client.stream("POST", "/v1/chat/completions", json=chat_payload) as stream_resp:
        assert stream_resp.status_code == 200
        sse_chunks = []
        saw_done = False
        for line in stream_resp.iter_lines():
            if line.startswith("data: "):
                payload_str = line[6:].strip()
                if payload_str == "[DONE]":
                    saw_done = True
                    break
                pkt = json.loads(payload_str)
                choices = pkt.get("choices", [])
                if choices and "delta" in choices[0] and "content" in choices[0]["delta"]:
                    sse_chunks.append(choices[0]["delta"]["content"])
        full_sse_text = "".join(sse_chunks)
        assert saw_done, "SSE did not terminate with [DONE]"
        print(f"✅ OpenAI SSE Streamed Output: '{full_sse_text.strip()}'")

    print("\n==================================================")
    print("🎉 ALL SERVER INTEGRATION TESTS PASSED 100%!")
    print("==================================================")

if __name__ == "__main__":
    test_endpoints()
