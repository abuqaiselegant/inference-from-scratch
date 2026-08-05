import tiktoken
import torch

from model import GPT


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Keep the smallest set of tokens whose cumulative probability
    is at least top_p (nucleus sampling).
    """
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be between 0 and 1.")

    # Sort logits from highest to lowest
    sorted_logits, sorted_indices = torch.sort(
        logits,
        descending=True,
        dim=-1,
    )

    # Convert sorted logits to probabilities
    sorted_probabilities = torch.softmax(
        sorted_logits,
        dim=-1,
    )

    # Cumulative probabilities
    cumulative_probabilities = torch.cumsum(
        sorted_probabilities,
        dim=-1,
    )

    # Remove tokens after the top_p boundary
    sorted_remove_mask = cumulative_probabilities > top_p

    # Keep the first token that crosses the threshold
    sorted_remove_mask[:, 1:] = sorted_remove_mask[:, :-1].clone()
    sorted_remove_mask[:, 0] = False

    # Map the mask back to original token order
    remove_mask = torch.zeros_like(
        logits,
        dtype=torch.bool,
    )

    remove_mask.scatter_(
        dim=-1,
        index=sorted_indices,
        src=sorted_remove_mask,
    )

    # Give removed tokens zero probability
    filtered_logits = logits.masked_fill(
        remove_mask,
        float("-inf"),
    )

    return filtered_logits


def select_next_token(
    next_token_logits: torch.Tensor,
    strategy: str = "sample",
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    """
    Select the next token using:
    - greedy decoding
    - temperature sampling
    - top-k sampling
    - top-p sampling
    - or a combination of top-k and top-p
    """

    # Greedy decoding
    if strategy == "greedy":
        return torch.argmax(
            next_token_logits,
            dim=-1,
            keepdim=True,
        )

    if strategy != "sample":
        raise ValueError(f"Unknown strategy: {strategy}")

    if temperature <= 0:
        raise ValueError(
            "Temperature must be greater than zero."
        )

    # --------------------------------------------------
    # Step 1: Temperature
    # --------------------------------------------------
    filtered_logits = next_token_logits / temperature

    # --------------------------------------------------
    # Step 2: Top-k filtering
    # --------------------------------------------------
    if top_k is not None:
        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        top_k = min(
            top_k,
            filtered_logits.size(-1),
        )

        top_values, _ = torch.topk(
            filtered_logits,
            k=top_k,
            dim=-1,
        )

        cutoff = top_values[:, -1:]

        filtered_logits = filtered_logits.masked_fill(
            filtered_logits < cutoff,
            float("-inf"),
        )

    # --------------------------------------------------
    # Step 3: Top-p filtering
    # --------------------------------------------------
    if top_p is not None:
        filtered_logits = apply_top_p(
            filtered_logits,
            top_p,
        )

    # --------------------------------------------------
    # Step 4: Probabilities
    # --------------------------------------------------
    probabilities = torch.softmax(
        filtered_logits,
        dim=-1,
    )

    # --------------------------------------------------
    # Step 5: Sample one token
    # --------------------------------------------------
    return torch.multinomial(
        probabilities,
        num_samples=1,
    )


def generate(
    model: GPT,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 40,
    strategy: str = "sample",
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    seed: int | None = None,
) -> str:
    """
    Generate text autoregressively.
    """

    # Make sampling reproducible
    if seed is not None:
        torch.manual_seed(seed)

    token_ids = tokenizer.encode(prompt)

    input_ids = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device,
    )

    for _ in range(max_new_tokens):
        # Keep sequence inside GPT-2 context length
        input_context = input_ids[
            :, -model.config.block_size:
        ]

        with torch.inference_mode():
            logits, _ = model(input_context)

        next_token_logits = logits[:, -1, :]

        next_token_id = select_next_token(
            next_token_logits=next_token_logits,
            strategy=strategy,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        token_id = next_token_id.item()

        input_ids = torch.cat(
            [input_ids, next_token_id],
            dim=1,
        )

        # EOS stopping
        if token_id == tokenizer.eot_token:
            break

    return tokenizer.decode(
        input_ids[0].tolist()
    )


def run_configuration(
    model,
    tokenizer,
    device,
    name: str,
    prompt: str,
    strategy: str,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
):
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    for run in range(3):
        output = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=40,
            strategy=strategy,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        print(f"\nRun {run + 1}:")
        print(output)


def main() -> None:
    device = get_device()
    print("Using device:", device)

    model = GPT.from_pretrained("gpt2")
    model.to(device)
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")

    prompt = "The future of artificial intelligence"

    # --------------------------------------------------
    # Greedy decoding
    # --------------------------------------------------
    run_configuration(
        model,
        tokenizer,
        device,
        name="Greedy Decoding",
        prompt=prompt,
        strategy="greedy",
    )

    # --------------------------------------------------
    # Temperature experiments
    # --------------------------------------------------
    run_configuration(
        model,
        tokenizer,
        device,
        name="Low Temperature (0.3)",
        prompt=prompt,
        strategy="sample",
        temperature=0.3,
    )

    run_configuration(
        model,
        tokenizer,
        device,
        name="Normal Temperature (1.0)",
        prompt=prompt,
        strategy="sample",
        temperature=1.0,
    )

    run_configuration(
        model,
        tokenizer,
        device,
        name="High Temperature (1.5)",
        prompt=prompt,
        strategy="sample",
        temperature=1.5,
    )

    # --------------------------------------------------
    # Top-k
    # --------------------------------------------------
    run_configuration(
        model,
        tokenizer,
        device,
        name="Top-k = 10",
        prompt=prompt,
        strategy="sample",
        temperature=1.0,
        top_k=10,
    )

    run_configuration(
        model,
        tokenizer,
        device,
        name="Top-k = 50",
        prompt=prompt,
        strategy="sample",
        temperature=1.0,
        top_k=50,
    )

    # --------------------------------------------------
    # Top-p
    # --------------------------------------------------
    run_configuration(
        model,
        tokenizer,
        device,
        name="Top-p = 0.9",
        prompt=prompt,
        strategy="sample",
        temperature=1.0,
        top_p=0.9,
    )

    run_configuration(
        model,
        tokenizer,
        device,
        name="Top-p = 0.95",
        prompt=prompt,
        strategy="sample",
        temperature=1.0,
        top_p=0.95,
    )

    # --------------------------------------------------
    # Combined Top-k + Top-p
    # --------------------------------------------------
    run_configuration(
        model,
        tokenizer,
        device,
        name="Top-k = 50 + Top-p = 0.9",
        prompt=prompt,
        strategy="sample",
        temperature=0.8,
        top_k=50,
        top_p=0.9,
    )

    # --------------------------------------------------
    # Seed reproducibility
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("Seed Reproducibility")
    print("=" * 60)

    for seed in [42, 42, 7]:
        output = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=40,
            strategy="sample",
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            seed=seed,
        )

        print(f"\nSeed {seed}:")
        print(output)


if __name__ == "__main__":
    main()