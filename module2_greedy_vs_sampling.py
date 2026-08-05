import tiktoken 
import torch

from model import GPT


def get_device():
    if torch.cuda.is_available():
        return "cude"
    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"

def select_next_token(next_token_logits, strategy, temperature =1.0):
    if strategy == "greedy":
            return torch.argmax(
                next_token_logits,
                dim = -1,
                keepdim = True,)
    
    if strategy == "sample":
        # convert logits into prob.
        if temperature <= 0:
            raise ValueError("Temperature must be greater than 0")
        
        scaled_logits = next_token_logits / temperature

        probabilites = torch.softmax(scaled_logits,dim = -1)

        # selecting one random probabilities.
        return torch.multinomial(probabilites, num_samples=1)
    
    raise ValueError(f"Unknown strategy: {strategy}")

def generate(model, tokenizer, prompt, device, max_new_tokens, strategy, temperature = 1.0):
    prompt_token_ids = tokenizer.encode(prompt)

    input_ids = torch.tensor(
          prompt_token_ids,
          dtype = torch.long,
          device = device,
     ).unsqueeze(0)

    for step in range(max_new_tokens):
        input_context = input_ids[:, -model.config.block_size:]
        with torch.inference_mode():
            logits ,  _ = model(input_context)
        next_token_logits = logits[:,-1,:]

        next_token_id = select_next_token(next_token_logits,strategy, temperature)

        token_id = next_token_id.item()

        input_ids = torch.cat([input_ids, next_token_id], dim = 1,)

        if token_id == tokenizer.eot_token:
            break
    generated_text = tokenizer.decode(input_ids[0].tolist())
    return generated_text


def main() -> None:
    device = get_device()
    print("Using device:", device)

    model = GPT.from_pretrained("gpt2")
    model.to(device)
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")

    prompt = "The future of artificial intelligence"

    temperatures = [0.2, 0.7, 1.0, 1.5]

    for temperature in temperatures:
        print("\n" + "=" * 60)
        print(f"TEMPERATURE: {temperature}")
        print("=" * 60)

        # Generate multiple outputs at each temperature
        for run in range(3):
            output = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                max_new_tokens=40,
                strategy="sample",
                temperature=temperature,
            )

            print(f"\nRun {run + 1}:")
            print(output)


if __name__ == "__main__":
    main()