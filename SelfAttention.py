import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        
        # Einsum does matrix multiplication for query*keys for each training example
        # with every other key
        attention = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        
        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-1e20"))
        
        attention = torch.softmax(attention / (self.embed_size ** (1 / 2)), dim=3)
        
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        
        out = self.fc_out(out)
        return out

# Example usage:
embed_size = 256  # Embedding size for each token
heads = 8  # Number of attention heads

# Create an instance of the model
self_attention = SelfAttention(embed_size, heads)

# Example input tensor with shape (batch_size, sequence_length, embed_size)
values = keys = query = torch.rand(1, 10, embed_size)

# Create a simple attention mask for the sequence (just for demonstration purposes)
mask = torch.ones(1, 10, 10)

# Forward pass through the model
output = self_attention(values, keys, query, mask)
print(output.shape)  # Expected output shape: (batch_size, sequence_length, embed_size)
