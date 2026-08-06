import math
import torch

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

torch.manual_seed(42)

batch_size = 1
sequence_length = 4
embedding_size = 6
head_size = 3

# --------------------------------------------------
# Input embeddings
# --------------------------------------------------

x = torch.randn(
    batch_size,
    sequence_length,
    embedding_size,
    device=device,
)

# --------------------------------------------------
# Projection matrices
# --------------------------------------------------

W_query = torch.randn(
    embedding_size,
    head_size,
    device=device,
)

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

# --------------------------------------------------
# Q, K, V
# --------------------------------------------------

queries = x @ W_query
keys = x @ W_key
values = x @ W_value

# --------------------------------------------------
# Attention scores
# --------------------------------------------------

scores = queries @ keys.transpose(-2, -1)
scores = scores / math.sqrt(head_size)

print("Original attention scores:\n", scores)

# --------------------------------------------------
# Create causal mask
# --------------------------------------------------

mask = torch.tril(
    torch.ones(
        sequence_length,
        sequence_length,
        device=device,
    )
)

print("\nCausal mask:\n", mask)

# --------------------------------------------------
# Apply the mask
# --------------------------------------------------

masked_scores = scores.masked_fill(
    mask == 0,
    float("-inf"),
)

print("\nMasked attention scores:\n", masked_scores)

# --------------------------------------------------
# Softmax
# --------------------------------------------------

attention_weights = torch.softmax(
    masked_scores,
    dim=-1,
)

print("\nAttention weights after masking:\n", attention_weights)

# --------------------------------------------------
# Verify row sums
# --------------------------------------------------

print("\nRow sums:\n", attention_weights.sum(dim=-1))

# --------------------------------------------------
# Context vectors
# --------------------------------------------------

context = attention_weights @ values

print("\nContext vectors:\n", context)
