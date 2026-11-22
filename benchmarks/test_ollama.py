import urllib.request
import json
import time

try:
    req = urllib.request.urlopen("http://localhost:11434/api/tags")
    models = json.loads(req.read().decode())['models']
    print("Ollama API is running! Models available:")
    for m in models:
        print(f" - {m['name']} ({m['size'] / (1024**3):.2f} GB)")
except Exception as e:
    print("Ollama API error:", e)
