import tiktoken
import torch

from model import GPT


def get_device() -> str:
    """Return the best available computation device."""
    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def generate_greedy(
    model: GPT,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 10,
) -> str:
    """
    Generate text by always selecting the token
    with the highest model score.
    """

    # Step 1: Convert the prompt into token IDs
    token_ids = tokenizer.encode(prompt)

    # Step 2: Create a batched tensor
    input_ids = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device,
    )

    print("\nOriginal prompt:", repr(prompt))
    print("Original token IDs:", token_ids)
    print("Starting shape:", input_ids.shape)

    # Step 3: Generate one token per loop iteration
    for step in range(max_new_tokens):
        # Keep only the supported context length
        input_context = input_ids[
            :, -model.config.block_size:
        ]

        # Step 4: Run the model
        with torch.inference_mode():
            logits, _ = model(input_context)

        # Step 5: Get scores for the next token
        next_token_logits = logits[:, -1, :]

        # Step 6: Select the highest-scoring token
        next_token_id = torch.argmax(
            next_token_logits,
            dim=-1,
            keepdim=True,
        )

        # Convert the token into readable text
        token_id = next_token_id.item()
        token_text = tokenizer.decode([token_id])

        print(
            f"Step {step + 1}:",
            f"id={token_id},",
            f"token={token_text!r}",
        )

        # Step 7: Append the token to the sequence
        input_ids = torch.cat(
            [input_ids, next_token_id],
            dim=1,
        )

        print("New input shape:", input_ids.shape)

        # Step 8: Stop if GPT-2 generates its end token
        if token_id == tokenizer.eot_token:
            print("End-of-text token generated.")
            break

    # Step 9: Convert the complete sequence back to text
    generated_token_ids = input_ids[0].tolist()
    generated_text = tokenizer.decode(generated_token_ids)

    return generated_text


def main() -> None:
    device = get_device()
    print("Using device:", device)

    model = GPT.from_pretrained("gpt2")
    model.to(device)
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")

    result = generate_greedy(
        model=model,
        tokenizer=tokenizer,
        prompt="AI systems can",
        device=device,
        max_new_tokens=10,
    )

    print("\nGenerated text:")
    print(result)


if __name__ == "__main__":
    main()