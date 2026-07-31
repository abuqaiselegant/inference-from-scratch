# AI Inference Journey

My hands-on notes and code while learning how LLM inference actually works —
building intuition one small module at a time.

## Modules

- `practice.py` — a single forward pass, dissected: prompt → token IDs → logits → next-token probabilities.
- `module0_inference.py` — device selection, timing, running the model.
- `module1_greedy_generation.py` — autoregressive greedy generation loop.

## Setup

```sh
python -m venv .venv
source .venv/bin/activate
pip install torch numpy transformers tiktoken
```

Then run any module, e.g.:

```sh
python module1_greedy_generation.py
```

## Roadmap

1. Sampling — temperature, top-k, top-p
2. KV cache — prefill vs decode, killing redundant recompute
3. Batching — throughput vs latency
4. Quantization — memory and precision trade-offs

## Credit

`model.py` is the GPT definition from Andrej Karpathy's
[nanoGPT](https://github.com/karpathy/nanoGPT) (MIT License), vendored here so
these scripts run standalone.
