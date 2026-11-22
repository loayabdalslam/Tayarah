import sys

packages = ['torch', 'transformers', 'llama_cpp', 'huggingface_hub', 'numpy', 'scipy', 'sympy']
for pkg in packages:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, '__version__', 'ok')
        print(f"{pkg}: {ver}")
    except ImportError as e:
        print(f"{pkg}: NOT INSTALLED ({e})")
