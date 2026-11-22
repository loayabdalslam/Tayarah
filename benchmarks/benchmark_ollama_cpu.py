import urllib.request
import json
import time

def test_ollama_cpu(model_name="gemma3:4b", prompt="Explain the theory of relativity in 2 sentences.", num_predict=32):
    req_data = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_gpu": 0,         # 100% CPU inference!
            "num_predict": num_predict,
            "temperature": 0.0
        }
    }
    
    t0 = time.time()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(req_data).encode(),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
    
    elapsed = time.time() - t0
    eval_count = data.get("eval_count", 0)
    eval_duration = data.get("eval_duration", 1) / 1e9 # convert ns to s
    tok_per_sec = eval_count / eval_duration if eval_duration > 0 else 0
    
    print(f"Model: {model_name} (100% CPU)")
    print(f"Response: {data.get('response', '').strip()}")
    print(f"Tokens generated: {eval_count}")
    print(f"Eval duration: {eval_duration:.3f} s")
    print(f"Baseline CPU Speed: {tok_per_sec:.2f} tokens/second")
    return tok_per_sec

if __name__ == "__main__":
    test_ollama_cpu()
