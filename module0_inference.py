import statistics
import time

import tiktoken
import torch

from model import GPT


# --------------------------------------------------
# Step 1: Select the available device
# --------------------------------------------------
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


# --------------------------------------------------
# Step 2: Wait for GPU operations to complete
# --------------------------------------------------
def synchronize_device(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()

    elif device == "mps":
        torch.mps.synchronize()


# --------------------------------------------------
# Step 3: Inspect one prompt in detail
# --------------------------------------------------
def inspect_prompt(
    model,
    tokenizer,
    prompt: str,
    device: str,
) -> None:
    # Convert text into token IDs
    token_ids = tokenizer.encode(prompt)

    print("\n" + "=" * 60)
    print("PROMPT INSPECTION")
    print("=" * 60)

    print("\nPrompt:")
    print(prompt)

    print("\nToken IDs:")
    print(token_ids)

    print("\nIndividual tokens:")

    for token_id in token_ids:
        token_text = tokenizer.decode([token_id])
        print(token_id, "->", repr(token_text))

    # Convert the Python list into a batched tensor
    input_ids = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device,
    )

    print("\nInput tensor:")
    print(input_ids)

    print("\nInput tensor shape:")
    print(input_ids.shape)

    # Wait for previous device operations to complete
    synchronize_device(device)

    start_time = time.perf_counter()

    # Run model inference
    with torch.inference_mode():
        logits, _ = model(input_ids)

    # Wait until model computation has completed
    synchronize_device(device)

    elapsed_time = time.perf_counter() - start_time

    print("\nLogits shape:")
    print(logits.shape)

    # Get vocabulary scores for the next token
    next_token_logits = logits[:, -1, :]

    print("\nNext-token logits shape:")
    print(next_token_logits.shape)

    # Convert logits into probabilities
    probabilities = torch.softmax(
        next_token_logits,
        dim=-1,
    )

    print("\nProbabilities shape:")
    print(probabilities.shape)

    print(
        "Probability sum:",
        probabilities.sum().item(),
    )

    # Get the five most likely token probabilities and IDs
    top_probabilities, top_token_ids = torch.topk(
        probabilities,
        k=5,
        dim=-1,
    )

    print("\nTop five predictions:")

    for probability, token_id in zip(
        top_probabilities[0],
        top_token_ids[0],
    ):
        predicted_text = tokenizer.decode(
            [token_id.item()]
        )

        print(
            repr(predicted_text),
            f"probability={probability.item():.4f}",
        )

    # Select the highest-scoring token
    next_token_id = torch.argmax(
        next_token_logits,
        dim=-1,
    )

    predicted_id = next_token_id.item()
    predicted_text = tokenizer.decode([predicted_id])

    print("\nGreedy predicted token ID:")
    print(predicted_id)

    print("\nGreedy predicted token:")
    print(repr(predicted_text))

    print("\nCompleted text:")
    print(prompt + predicted_text)

    print(f"\nInference time: {elapsed_time:.6f} seconds")


# --------------------------------------------------
# Step 4: Benchmark one prompt multiple times
# --------------------------------------------------
def benchmark_prompt(
    model,
    tokenizer,
    prompt: str,
    device: str,
    warmup_runs: int = 3,
    measured_runs: int = 10,
) -> None:
    token_ids = tokenizer.encode(prompt)

    input_ids = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device,
    )

    # Warm-up runs are not measured
    with torch.inference_mode():
        for _ in range(warmup_runs):
            model(input_ids)

    synchronize_device(device)

    inference_times = []

    # Measure the same prompt several times
    with torch.inference_mode():
        for _ in range(measured_runs):
            synchronize_device(device)

            start_time = time.perf_counter()

            logits, _ = model(input_ids)

            synchronize_device(device)

            elapsed_time = time.perf_counter() - start_time
            inference_times.append(elapsed_time)

    # Find the highest-probability next token
    next_token_logits = logits[:, -1, :]

    next_token_id = torch.argmax(
        next_token_logits,
        dim=-1,
    ).item()

    predicted_text = tokenizer.decode([next_token_id])

    print("\n" + "-" * 60)
    print("Prompt:", repr(prompt))
    print("Token count:", len(token_ids))
    print("Top token:", repr(predicted_text))

    print(
        f"Average time: "
        f"{statistics.mean(inference_times):.6f} seconds"
    )

    print(
        f"Median time:  "
        f"{statistics.median(inference_times):.6f} seconds"
    )

    print(
        f"Minimum time: "
        f"{min(inference_times):.6f} seconds"
    )

    print(
        f"Maximum time: "
        f"{max(inference_times):.6f} seconds"
    )


# --------------------------------------------------
# Step 5: Main program
# --------------------------------------------------
def main() -> None:
    # Select the device
    device = get_device()
    print("Using device:", device)

    # Load the model only once
    model = GPT.from_pretrained("gpt2")
    model.to(device)
    model.eval()

    # Create the tokenizer only once
    tokenizer = tiktoken.get_encoding("gpt2")

    # Inspect one prompt in detail
    inspect_prompt(
        model=model,
        tokenizer=tokenizer,
        prompt="AI systems can",
        device=device,
    )

    # Benchmark prompts with different token lengths
    prompts = [
        "AI is",
        "AI systems can",
        "Artificial intelligence systems can",
        (
            "Artificial intelligence systems are increasingly "
            "being used in production applications because they can"
        ),
    ]

    print("\n" + "=" * 60)
    print("PROMPT BENCHMARK")
    print("=" * 60)

    for prompt in prompts:
        benchmark_prompt(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
        )


if __name__ == "__main__":
    main()