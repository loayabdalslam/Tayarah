import urllib.request
import json
import time

def benchmark_baseline(prompt="List the planets of the solar system in order from the sun.", n_predict=40):
    url = "http://127.0.0.1:8085/completion"
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "stream": False
    }
    
    t0 = time.time()
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode())
    
    total_time = time.time() - t0
    content = res.get("content", "")
    tokens_evaluated = res.get("tokens_evaluated", 0)
    tokens_predicted = res.get("tokens_predicted", 0)
    
    # Timing details from llama-server
    timings = res.get("timings", {})
    predicted_per_second = timings.get("predicted_per_second", 0)
    predicted_ms = timings.get("predicted_ms", 0)
    
    print("=== BASELINE CPU INFERENCE (4.97GB GGUF on Core i7 CPU) ===")
    print(f"Generated text: {content.strip()}")
    print(f"Tokens predicted: {tokens_predicted}")
    print(f"Predict time: {predicted_ms:.2f} ms")
    print(f"Predicted speed: {predicted_per_second:.2f} tokens/second")
    print(f"Total roundtrip time: {total_time:.3f} s")
    return predicted_per_second, content

if __name__ == "__main__":
    benchmark_baseline()
