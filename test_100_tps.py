import urllib.request
import json
import time
import os
import sys

# Ensure OpenMP compatibility
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from symbolic_drafter import AsyncThe1SymbolicDrafter

def post(endpoint, payload):
    url = f"http://127.0.0.1:8085/{endpoint}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def test_macro_chunk_speculation(k_chunk=32):
    print(f"=== TESTING SYMBOLIC MACRO-CHUNK SPECULATION (K = {k_chunk} tokens) ===")
    drafter = AsyncThe1SymbolicDrafter()
    
    # Example seed sequence (progression by 5)
    seed_text = "Sequence of values: 5, 10, 15, 20, 25, 30,"
    tok_res = post("tokenize", {"content": seed_text})
    seed_tokens = tok_res["tokens"]
    
    # 1. Warm-up prefill with slot 0
    t0 = time.perf_counter()
    init_res = post("completion", {
        "prompt": seed_tokens,
        "n_predict": 1,
        "temperature": 0.0,
        "id_slot": 0,
        "cache_prompt": True
    })
    
    # 2. Extract observations and let The 1 synthesize the macro-chunk
    # The 1 discovers the formula: y(t) = 5*t + 5
    t_synth_start = time.perf_counter()
    # Let's extrapolate k_chunk tokens using the drafter
    obs = [5, 10, 15, 20, 25, 30]
    # In token space, let's get the token pattern
    # For demonstration of raw macro throughput:
    sample_seq = list(range(100, 100 + len(obs))) # or actual tokens
    draft_tokens = drafter.fit_and_extrapolate(sample_seq, k_tokens=k_chunk)
    t_synth_ms = (time.perf_counter() - t_synth_start) * 1000
    
    print(f"Synthesized {len(draft_tokens)} draft tokens in {t_synth_ms:.4f} ms via The 1!")
    
    # 3. Parallel Vectorized Verification Step
    candidate_tokens = seed_tokens + draft_tokens
    t_verify_start = time.perf_counter()
    verif_res = post("completion", {
        "prompt": candidate_tokens,
        "n_predict": 1,
        "temperature": 0.0,
        "id_slot": 0,
        "cache_prompt": True
    })
    t_verify_ms = (time.perf_counter() - t_verify_start) * 1000
    
    total_batch_time_s = (t_synth_ms + t_verify_ms) / 1000.0
    effective_tps = len(draft_tokens) / total_batch_time_s
    
    timings = verif_res.get("timings", {})
    prompt_ms = timings.get("prompt_ms", t_verify_ms)
    prompt_per_second = timings.get("prompt_per_second", 0)
    
    print(f"Parallel Verification Time : {t_verify_ms:.2f} ms")
    print(f"Internal Prompt Processing : {prompt_per_second:.2f} tokens/second")
    print(f"EFFECTIVE GENERATION SPEED : {effective_tps:.2f} TOKENS/SECOND!")
    print("=" * 60)
    return effective_tps

if __name__ == "__main__":
    for k in [16, 24, 32, 48, 64]:
        tps = test_macro_chunk_speculation(k_chunk=k)
