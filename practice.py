import time

import tiktoken
import torch

from model import GPT




# to check which devixe
def get_device() :
    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"

device = get_device()
# print("using device: ", device)

model = GPT.from_pretrained("gpt2")

model.to(device)
model.eval()


tokenizer = tiktoken.get_encoding("gpt2")
prompt = "AI systems can"
token_ids =  tokenizer.encode(prompt)

print("\nPrompt:")
print(prompt)

print("\nToken IDs:")
print(token_ids)

print("\nIndividual tokens:")

for token_id in token_ids:
    token_text = tokenizer.decode([token_id])
    print(token_id, "->", repr(token_text))

print(type(token_ids))
input_ids = torch.tensor(
    token_ids,
    dtype=torch.long,
    device=device,
)

input_ids = input_ids.unsqueeze(0)

print("\nInput tensor:")
print(input_ids)

print("\nInput tensor shape:")
print(input_ids.shape)


start_time = time.perf_counter()

with torch.inference_mode():
    logits, _ = model(input_ids)

elapsed_time = time.perf_counter() - start_time

print("\nLogits shape:")
print(logits.shape)

next_token_logits = logits[:, -1, :]
next_token_id = torch.argmax(
    next_token_logits,
    dim=-1,
)
print("\nnext token logits shape:")
print(next_token_logits.shape)

probabilities = torch.softmax(
    next_token_logits,
    dim=-1,
)
print(probabilities.shape)

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
    predicted_text = tokenizer.decode([token_id.item()])

    print(
        repr(predicted_text),
        f"probability: {probability.item():.4f}",
    )
# print(top_probabilities)
# for i in top_probabilities[0]:
#     print(tokenizer.decode([i]))

predicted_id = next_token_id.item()

predicted_text = tokenizer.decode([predicted_id])

print("\nPredicted token ID:")
print(predicted_id)

print("\nPredicted token:")
print(repr(predicted_text))

print("\nCompleted text:")
print(prompt + predicted_text)

print(f"\nInference time: {elapsed_time:.4f} seconds")

# print(logits[:,:,:])