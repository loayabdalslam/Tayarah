import os
import json
import time
from symb_runtime import SymbLLMRuntime

def run_benchmarks():
    runtime = SymbLLMRuntime(base_url="http://127.0.0.1:8085", draft_k=4)

    test_prompts = [
        {
            "name": "Case 1: Mathematical Progression & Dynamic Sequences",
            "prompt": "Continue the numerical sequence: 10, 20, 30, 40, 50, 60,",
            "max_tokens": 30
        },
        {
            "name": "Case 2: Structured Code & Iterative Functions",
            "prompt": "Write Python array values for powers of two starting at 1: [1, 2, 4, 8, 16,",
            "max_tokens": 30
        },
        {
            "name": "Case 3: Structured Data Coordinates (Linear Manifold)",
            "prompt": "Coordinates following y = 2*x + 1: (0, 1), (1, 3), (2, 5), (3, 7),",
            "max_tokens": 30
        },
        {
            "name": "Case 4: Multi-Step Scientific Reasoning",
            "prompt": "Explain the laws of motion by Isaac Newton step by step: Step 1:",
            "max_tokens": 35
        }
    ]

    results = []

    print("=" * 80)
    print("      SYMBLLM (racore/The1) vs BASELINE CPU RUNTIME BENCHMARK")
    print("      Model: Gemma-4B/7B-class Q4_K_M (4.97 GB) on Intel Core i7-13620H CPU")
    print("=" * 80)

    for i, test in enumerate(test_prompts, 1):
        print(f"\n[{i}/{len(test_prompts)}] Running {test['name']}...")
        
        # 1. Baseline Run
        print("  -> Executing Baseline (Standard Autoregressive CPU)...")
        b_res = runtime.generate_baseline(test["prompt"], max_tokens=test["max_tokens"])
        
        # 2. SymbLLM Run
        print("  -> Executing SymbLLM (Symbolic Drafter via racore/The1)...")
        s_res = runtime.generate_symbolic_speculative(test["prompt"], max_tokens=test["max_tokens"])

        # Speedup calculation
        b_speed = b_res["tokens_per_second"]
        s_speed = s_res["tokens_per_second"]
        speedup = s_speed / b_speed if b_speed > 0 else 1.0

        item = {
            "test_name": test["name"],
            "prompt": test["prompt"],
            "baseline_speed": b_speed,
            "symbllm_speed": s_speed,
            "speedup_ratio": round(speedup, 2),
            "baseline_time_ms": b_res["total_time_ms"],
            "symbllm_time_ms": s_res["total_time_ms"],
            "baseline_output": b_res["output"],
            "symbllm_output": s_res["output"],
            "acceptance_ratio": s_res["acceptance_ratio"],
            "draft_latency": s_res["draft_eval_latency"]
        }
        results.append(item)

        print(f"     * Baseline Speed : {b_speed:.2f} tok/s  ({b_res['total_time_ms']:.1f} ms)")
        print(f"     * SymbLLM Speed  : {s_speed:.2f} tok/s  ({s_res['total_time_ms']:.1f} ms)")
        print(f"     * Speedup Factor : {speedup:.2f}x FASTER!")
        print(f"     * Spec Acceptance: {s_res['acceptance_ratio']}")
        print(f"     * Draft Latency  : {s_res['draft_eval_latency']}")

    # Summary
    print("\n" + "=" * 80)
    print("                     FINAL BENCHMARK SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Benchmark Test Case':<45} | {'Baseline':<10} | {'SymbLLM':<10} | {'Speedup':<8}")
    print("-" * 80)
    avg_speedup = 0.0
    for r in results:
        print(f"{r['test_name'][:43]:<45} | {r['baseline_speed']:>6.2f} t/s | {r['symbllm_speed']:>6.2f} t/s | {r['speedup_ratio']:>6.2f}x")
        avg_speedup += r['speedup_ratio']
    avg_speedup /= len(results)
    print("-" * 80)
    print(f"{'AVERAGE OVERALL SPEEDUP':<45} | {'':<10} | {'':<10} | {avg_speedup:>6.2f}x")
    print("=" * 80)

    # Save to JSON
    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump({"summary": results, "avg_speedup": round(avg_speedup, 2)}, f, indent=2, ensure_ascii=False)
    print("\nResults successfully saved to benchmark_results.json")

if __name__ == "__main__":
    run_benchmarks()
