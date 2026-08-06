import math
import torch

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Device: ", device)

torch.manual_seed(3)

batch_size = 1
sequence_length = 4
embedding_size = 6
head_size = 8

# creating artificial token embeddings
x = torch.randn(
    batch_size,
    sequence_length, 
    embedding_size,
    device = device,
)

print("x: ",x)
print("x shape", x.shape)


# calculating projection matrices
W_query = torch.randn(
    embedding_size, 
    head_size,
    device = device,
)

print("w_query: ",W_query)
print("w query", W_query.shape)

W_key = torch.randn(
    embedding_size,
    head_size,
    device=device,
)

W_value = torch.randn(
    embedding_size,
    head_size,
    device=device,
)

# calculate q,k, v tensors... [1, 4, 6] @ [6, 3] → [1, 4, 3]

queries = x @ W_query
keys = x @ W_key
values = x @ W_value

print("queries : ",queries)
print("queries shape : ", queries.shape)
# comparing every query with every key
# --------------------------------------------------
# Step 7: Compare every query with every key
#
# queries:                  [1, 4, 3]
# keys.transpose(-2, -1):  [1, 3, 4]
#
# Result:                   [1, 4, 4]
# --------------------------------------------------

attention_scores = queries @ keys.transpose(-2, -1)

print("attention_scores: ", attention_scores)
print("attention_shape: ", attention_scores.shape)


scaled_scores = attention_scores / math.sqrt(head_size)

print("\nScaled attention scores:\n", scaled_scores)


# convert score in probabilities
attention_weights = torch.softmax(
    scaled_scores,
    dim=-1,
)

print("\nAttention weights:\n", attention_weights)
print("Attention weights shape", attention_weights.shape)


row_sums = attention_weights.sum(dim=-1)

print("\nAttention row sums:\n", row_sums)


# --------------------------------------------------
# Step 11: Combine the Value vectors
#
# attention_weights: [1, 4, 4]
# values:             [1, 4, 3]
#
# Result:             [1, 4, 3]
# --------------------------------------------------

context = attention_weights @ values

print("\nContext shape:", context.shape)
print("Contextual token representations:\n", context)

first_token_weights = attention_weights[0, 0]
print("first_token_weights: ", first_token_weights)
print("first_token_weights shape: ", first_token_weights.shape)