import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(d_model, d_model)
        
    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections in all three directions
        query = self.linear_q(query)
        key = self.linear_k(key)
        value = self.linear_v(value)
        
        # Split into multiple heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        
        # Scaled dot product attention
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, value)
        
        # Concatenate attention heads and put through final linear layer
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        
        output = self.linear_out(attn_output)
        return output, attn_weights

# Example usage:
d_model = 512  # Model dimension
num_heads = 8   # Number of attention heads

# Create an instance of the model
multi_head_attention = MultiHeadAttention(d_model, num_heads)

# Example input tensor with shape (batch_size, sequence_length, d_model)
query = key = value = torch.rand(1, 60, d_model)

# Create a simple attention mask for the sequence (just for demonstration purposes)
mask = torch.zeros(1, 60, 60).bool()

# Forward pass through the model
output, attn_weights = multi_head_attention(query, key, value, mask)
print(output.shape)  # Expected output shape: (batch_size, sequence_length, d_model)
print(attn_weights.shape)  # Expected attention weights shape: (batch_size, num_heads, sequence_length, sequence_length)
