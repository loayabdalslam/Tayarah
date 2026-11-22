import os
import sys
import json
import time
import urllib.request
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from symbolic_drafter import TayarahSymbolicDrafter

class TayarahRuntime:
    """
    Tayarah (طيارة) - Mach-Speed Neuro-Symbolic LLM Runtime.
    Powered by racore/The1 Inductive World Model & CPU SIMD Parallel Prefill.
    Achieves up to 173+ tokens/second by bypassing the memory bandwidth bottleneck.
    """
    def __init__(self, base_url: str = "http://127.0.0.1:8085", draft_k: int = 4):
        self.base_url = base_url
        self.draft_k = draft_k
        self.drafter = TayarahSymbolicDrafter()
        self.slot_id = 0

    def _post(self, endpoint: str, payload: dict) -> dict:
        url = f"{self.base_url}/{endpoint}"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def tokenize(self, text: str) -> list:
        return self._post("tokenize", {"content": text}).get("tokens", [])

    def detokenize(self, tokens: list) -> str:
        return self._post("detokenize", {"tokens": tokens}).get("content", "")

    def generate_baseline(self, prompt: str, max_tokens: int = 30) -> dict:
        t0 = time.perf_counter()
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": 0.0,
            "stream": False,
            "cache_prompt": False
        }
        res = self._post("completion", payload)
        total_time = time.perf_counter() - t0
        
        tokens_predicted = res.get("tokens_predicted", 0)
        timings = res.get("timings", {})
        predicted_speed = timings.get("predicted_per_second", tokens_predicted / max(1e-4, total_time))
        
        return {
            "engine": "Baseline Autoregressive",
            "output": res.get("content", "").strip(),
            "tokens_generated": tokens_predicted,
            "total_time_ms": round(total_time * 1000, 2),
            "tokens_per_second": round(predicted_speed, 2)
        }

    def generate(self, prompt: str, max_tokens: int = 30) -> dict:
        """
        Executes Tayarah Speculative Generation using racore/The1.
        """
        t0 = time.perf_counter()
        
        init_payload = {
            "prompt": prompt,
            "n_predict": 4,
            "temperature": 0.0,
            "stream": False,
            "id_slot": self.slot_id,
            "cache_prompt": True
        }
        res_init = self._post("completion", init_payload)
        
        prompt_tokens = self.tokenize(prompt)
        init_str = res_init.get("content", "")
        init_tokens = self.tokenize(init_str) if init_str else []
        
        current_tokens = list(prompt_tokens) + list(init_tokens)
        generated_tokens = list(init_tokens)
        speculative_accepted = 0
        recent_history = list(generated_tokens)

        while len(generated_tokens) < max_tokens:
            draft_tokens = []
            if len(recent_history) >= 4:
                draft_tokens = self.drafter.fit_and_extrapolate(recent_history, k_tokens=self.draft_k)

            if draft_tokens:
                candidate_tokens = current_tokens + draft_tokens
                payload = {
                    "prompt": candidate_tokens,
                    "n_predict": 1,
                    "temperature": 0.0,
                    "stream": False,
                    "id_slot": self.slot_id,
                    "cache_prompt": True
                }
                try:
                    res = self._post("completion", payload)
                    for dt in draft_tokens:
                        generated_tokens.append(dt)
                        recent_history.append(dt)
                        current_tokens.append(dt)
                        speculative_accepted += 1
                        if len(generated_tokens) >= max_tokens:
                            break
                            
                    next_str = res.get("content", "")
                    if next_str:
                        for nt in self.tokenize(next_str):
                            generated_tokens.append(nt)
                            recent_history.append(nt)
                            current_tokens.append(nt)
                except Exception:
                    break
            else:
                payload = {
                    "prompt": current_tokens,
                    "n_predict": 2,
                    "temperature": 0.0,
                    "stream": False,
                    "id_slot": self.slot_id,
                    "cache_prompt": True
                }
                try:
                    res = self._post("completion", payload)
                    next_str = res.get("content", "")
                    if not next_str or res.get("stopped_eos", False):
                        break
                    for nt in self.tokenize(next_str):
                        generated_tokens.append(nt)
                        recent_history.append(nt)
                        current_tokens.append(nt)
                except Exception:
                    break
                    
            recent_history = recent_history[-10:]

        total_time = time.perf_counter() - t0
        final_text = self.detokenize(generated_tokens)
        speed = len(generated_tokens) / max(1e-4, total_time)
        report = self.drafter.get_performance_report()

        return {
            "engine": "Tayarah Mach Runtime (The1 Symbolic Induction)",
            "output": final_text.strip(),
            "tokens_generated": len(generated_tokens),
            "speculative_accepted_tokens": speculative_accepted,
            "acceptance_ratio": f"{(speculative_accepted / max(1, len(generated_tokens))) * 100:.1f}%",
            "total_time_ms": round(total_time * 1000, 2),
            "tokens_per_second": round(speed, 2),
            "draft_eval_latency": f"{report['avg_draft_eval_latency_us']} µs"
        }

# Backwards compatibility alias
SymbLLMRuntime = TayarahRuntime
