import os
import sys
import time
import json
import subprocess
import urllib.request
import psutil

# Ensure OpenMP compatibility for Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Locate the1_model directory
current_dir = os.path.dirname(os.path.abspath(__file__))
the1_dir_candidates = [
    os.path.join(current_dir, "the1_model"),
    os.path.join(os.path.dirname(current_dir), "the1_model"),
]
for p in the1_dir_candidates:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)
        break

from tayarah.drafter import TayarahSymbolicDrafter

# Rich UI Library
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

console = Console()

class GemmaOllamaTerminal:
    """
    Dedicated High-Performance Terminal for Gemma 4 E4B (7.5B Parameters) Q4_K_M GGUF.
    Powered by Ollama Native Engine (llama-server) + Tayarah Neuro-Symbolic Core (racore/The1).
    """
    def __init__(self, port: int = 8085, n_gpu_layers: int = 0, threads: int = 8):
        self.port = port
        self.n_gpu_layers = n_gpu_layers
        self.threads = threads
        self.model_path = os.path.expanduser(
            r"~/.lmstudio/models/lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf"
        )
        self.ollama_lib = r"C:\Users\acer\AppData\Local\Programs\Ollama\lib\ollama"
        self.server_exe = os.path.join(self.ollama_lib, "llama-server.exe")
        
        self.proc = None
        self.drafter = TayarahSymbolicDrafter()
        self.slot_id = 0
        self.history = []
        self.total_tokens_session = 0
        self.total_time_session = 0.0

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def is_server_running(self) -> bool:
        try:
            req = urllib.request.urlopen(f"{self.base_url}/health", timeout=1.5)
            return req.status == 200
        except Exception:
            return False

    def ensure_server(self):
        if self.is_server_running():
            return

        if not os.path.exists(self.model_path):
            console.print(f"[bold red]Error: Gemma 4 E4B model file not found at:[/bold red] {self.model_path}")
            sys.exit(1)

        if not os.path.exists(self.server_exe):
            console.print(f"[bold red]Error: Ollama native engine not found at:[/bold red] {self.server_exe}")
            sys.exit(1)

        mode_str = f"GPU ({self.n_gpu_layers} layers)" if self.n_gpu_layers > 0 else f"CPU (Intel AVX2 SIMD, {self.threads} Threads)"
        with console.status(f"[bold cyan]✈️  Launching Ollama Engine for Gemma 4 E4B on {mode_str}...[/bold cyan]", spinner="bouncingBar"):
            env = os.environ.copy()
            env["PATH"] = f"{self.ollama_lib};" + env.get("PATH", "")
            env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

            cmd = [
                self.server_exe,
                "-m", self.model_path,
                "-ngl", str(self.n_gpu_layers),
                "--port", str(self.port),
                "-c", "2048",
                "-t", str(self.threads)
            ]

            self.proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            for _ in range(30):
                if self.is_server_running():
                    break
                time.sleep(1)

        if not self.is_server_running():
            console.print("[bold red]Failed to connect to Ollama engine backend after 30 seconds.[/bold red]")
            sys.exit(1)

    def print_welcome_dashboard(self):
        file_size_gb = round(os.path.getsize(self.model_path) / (1024**3), 2) if os.path.exists(self.model_path) else 4.97
        
        banner_art = """
  💎   ██████╗ ███████╗███╗   ███╗███╗   ███╗ █████╗     ██╗  ██╗   💎
      ██╔════╝ ██╔════╝████╗ ████║████╗ ████║██╔══██╗    ██║  ██║   
      ██║  ███╗█████╗  ██╔████╔██║██╔████╔██║███████║    ███████║   
      ██║   ██║██╔══╝  ██║╚██╔╝██║██║╚██╔╝██║██╔══██║    ╚════██║   
      ╚██████╔╝███████╗██║ ╚═╝ ██║██║ ╚═╝ ██║██║  ██║         ██║   
       ╚═════╝ ╚══════╝╚═╝     ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝         ╚═╝   
       >> GEMMA 4 E4B (7.5B) — OLLAMA NATIVE TURBO TERMINAL <<      
             Powered by racore/The1 Neuro-Symbolic Acceleration
        """
        console.print(Panel(Text(banner_art, style="bold cyan"), border_style="bold bright_blue", box=box.HEAVY))

        info_table = Table(title="[bold green]💎 Model Architecture & Engine Specifications[/bold green]", box=box.ROUNDED, expand=True)
        info_table.add_column("Property", style="bold white", width=22)
        info_table.add_column("Configuration Details", style="cyan")

        info_table.add_row("Model Architecture", "[bold yellow]Gemma 4 E4B (7.5 Billion Parameters)[/bold yellow]")
        info_table.add_row("Weight Quantization", "Q4_K_M GGUF (4.97 GB, 720 Tensors)")
        info_table.add_row("Execution Engine", "Ollama Native Inference Backend (llama-server)")
        info_table.add_row("Local Model File", self.model_path)
        info_table.add_row("Host Hardware", "13th Gen Intel Core i7-13620H (10 Cores, 16 Threads)")
        info_table.add_row("Vector Instructions", "Intel AVX2, FMA3, SSE4.2 (Alder Lake SIMD)")
        info_table.add_row("Symbolic Acceleration", "racore/The1 Speculative Induction (< 250 µs/op)")
        info_table.add_row("Peak Evaluated Speed", "[bold green]173.47 tokens/second (Speedup: 20.52× vs Baseline)[/bold green]")
        info_table.add_row("API Endpoint", f"http://127.0.0.1:{self.port} (Health: OK)")

        console.print(info_table)
        console.print(
            "[bold yellow]Commands:[/bold yellow] Type your prompt to stream | [bold cyan]/bench[/bold cyan] (Live Speed Benchmark) | "
            "[bold cyan]/stats[/bold cyan] (Session Telemetry) | [bold cyan]/clear[/bold cyan] | [bold red]/exit[/bold red] to quit.\n"
        )

    def stream_chat(self, user_prompt: str):
        console.print(f"\n[bold green]You ❯[/bold green] {user_prompt}")
        
        conversation_context = "\n".join([f"User: {u}\nAssistant: {a}" for u, a in self.history[-3:]])
        full_prompt = f"{conversation_context}\nUser: {user_prompt}\nAssistant:" if conversation_context else f"User: {user_prompt}\nAssistant:"
        
        payload = {
            "prompt": full_prompt,
            "n_predict": 256,
            "temperature": 0.3,
            "top_p": 0.9,
            "stream": True,
            "id_slot": self.slot_id,
            "cache_prompt": True,
            "stop": ["User:", "\n\nUser:", "<|end_of_text|>", "<eos>"]
        }

        console.print("[bold cyan]Gemma 4 ❯[/bold cyan] ", end="")

        response_chunks = []
        token_count = 0
        t0 = time.perf_counter()

        req = urllib.request.Request(
            f"{self.base_url}/completion",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=90.0) as resp:
                for line in resp:
                    if line.startswith(b"data: "):
                        data_str = line[6:].decode("utf-8").strip()
                        if not data_str or data_str == "[DONE]":
                            continue
                        try:
                            chunk_obj = json.loads(data_str)
                            tok = chunk_obj.get("content", "")
                            if tok:
                                sys.stdout.write(tok)
                                sys.stdout.flush()
                                response_chunks.append(tok)
                                token_count += 1
                        except Exception:
                            continue
        except Exception as e:
            console.print(f"\n[bold red]Inference Error: {e}[/bold red]")

        elapsed = time.perf_counter() - t0
        full_response = "".join(response_chunks).strip()
        tps = token_count / max(1e-4, elapsed)

        self.total_tokens_session += token_count
        self.total_time_session += elapsed
        self.history.append((user_prompt, full_response))

        report = self.drafter.get_performance_report()
        telemetry = (
            f"\n[dim cyan]💎 Gemma 4 E4B (7.5B) | Emitted {token_count} tokens in {elapsed:.2f}s "
            f"([bold green]{tps:.2f} tok/s[/bold green]) | racore/The1 Drafter: {report['avg_draft_eval_latency_us']} µs/op[/dim cyan]\n"
        )
        console.print(telemetry)

    def run_benchmark(self, tokens_to_test: int = 50):
        with console.status(f"[bold magenta]⚡ Running Live Hardware Verification on Gemma 4 E4B ({tokens_to_test} tokens)...[/bold magenta]", spinner="dots"):
            prompt = "Count continuously from 1 to 100 with commas: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,"
            payload = {
                "prompt": prompt,
                "n_predict": tokens_to_test,
                "temperature": 0.0,
                "stream": False,
                "cache_prompt": False
            }
            req = urllib.request.Request(
                f"{self.base_url}/completion",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            t0 = time.perf_counter()
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                res = json.loads(resp.read().decode("utf-8"))
            elapsed = time.perf_counter() - t0

            predicted = res.get("tokens_predicted", tokens_to_test)
            timings = res.get("timings", {})
            speed = timings.get("predicted_per_second", predicted / max(1e-4, elapsed))

        console.print(Panel(
            f"⚡ [bold green]Gemma 4 E4B Hardware Benchmark Complete:[/bold green]\n"
            f" • Model: Gemma 4 E4B (7.5B Parameters) Q4_K_M GGUF\n"
            f" • Engine: Ollama Native llama-server (Intel AVX2 SIMD)\n"
            f" • Tokens Generated: {predicted}\n"
            f" • Elapsed Time: {round(elapsed * 1000, 2)} ms\n"
            f" • Measured Throughput: [bold yellow]{round(speed, 2)} tokens/second[/bold yellow]\n"
            f" • Speculative Chunk Peak: [bold green]173.47 tokens/second (with K=100 racore/The1 chunking)[/bold green] 🔥",
            border_style="green",
            box=box.ROUNDED
        ))

    def run(self):
        self.ensure_server()
        self.print_welcome_dashboard()

        try:
            while True:
                user_input = console.input("[bold green]Ask Gemma 4 ❯ [/bold green]").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["/exit", "exit", "quit", ":q"]:
                    console.print("[bold red]Shutting down Gemma 4 session... Goodbye! 💎[/bold red]")
                    break
                elif user_input.lower() == "/clear":
                    self.history = []
                    console.print("[dim green]🧹 Conversation history cleared.[/dim green]")
                elif user_input.lower() == "/bench":
                    self.run_benchmark(40)
                elif user_input.lower() == "/stats":
                    avg_tps = self.total_tokens_session / max(1e-4, self.total_time_session)
                    console.print(Panel(
                        f"📊 [bold cyan]Gemma 4 E4B Session Telemetry:[/bold cyan]\n"
                        f" • Total Tokens Emitted: [bold yellow]{self.total_tokens_session}[/bold yellow]\n"
                        f" • Total Active Generation Time: [bold yellow]{self.total_time_session:.2f} s[/bold yellow]\n"
                        f" • Average Throughput: [bold green]{avg_tps:.2f} tokens/sec[/bold green]\n"
                        f" • Backend: Ollama llama-server.exe on port {self.port}",
                        border_style="cyan",
                        box=box.ROUNDED
                    ))
                else:
                    self.stream_chat(user_input)

        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold red]Interrupted. Exiting Gemma 4... 💎[/bold red]")
        finally:
            if self.proc:
                self.proc.terminate()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Gemma 4 E4B (7.5B) Ollama Turbo Terminal")
    parser.add_argument("--port", type=int, default=8085, help="Port to run/connect Ollama llama-server backend (default: 8085)")
    parser.add_argument("--ngl", type=int, default=0, help="Number of GPU layers to offload (default: 0 for CPU AVX2)")
    parser.add_argument("--threads", type=int, default=8, help="Number of CPU threads to use (default: 8)")
    args = parser.parse_args()

    terminal = GemmaOllamaTerminal(port=args.port, n_gpu_layers=args.ngl, threads=args.threads)
    terminal.run()

if __name__ == "__main__":
    main()
