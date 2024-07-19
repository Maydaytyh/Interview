import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleHeadAttention(nn.Module):
    def __init__(self, d_model):
        super(SingleHeadAttention, self).__init__()
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        # Linear transformations of query, key, and value
        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)
        
        # Compute the attention scores (dot product of query and key)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (key.size(-1) ** 0.5)
        
        # Apply the mask if it is provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        # Compute the attention weights using softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply the attention weights to the value
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights

# Example usage:
d_model = 512  # Model dimension

# Create an instance of the model
single_head_attention = SingleHeadAttention(d_model)

# Example input tensor with shape (batch_size, sequence_length, d_model)
query = key = value = torch.rand(1, 60, d_model)

# Create a simple attention mask for the sequence (just for demonstration purposes)
mask = torch.zeros(1, 60, 60).bool()

# Forward pass through the model
output, attention_weights = single_head_attention(query, key, value, mask)
print(output.shape)  # Expected output shape: (batch_size, sequence_length, d_model)
print(attention_weights.shape)  # Expected attention weights shape: (batch_size, sequence_length, sequence_length)
