import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads
        
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        
        # Adjusted output dimension of linear layer
        self.out = nn.Linear(d_model, d_model)
    
    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        
        q = self.split_heads(self.Wq(x), batch_size)
        print(q.shape)
        k = self.split_heads(self.Wk(x), batch_size)
        print(k.shape)
        v = self.split_heads(self.Wv(x), batch_size)
        print(v.shape)
        
        q = q / (self.depth ** 0.5)
        
        scores = torch.matmul(q, k.permute(0, 1, 3, 2))
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)
        
        out = torch.matmul(attention_weights, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, -1)
        
        # Apply adjusted linear layer
        out = self.out(out)
        
        return out

class FeedForward(nn.Module):
    def __init__(self, d_model, dff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, dff)
        self.linear2 = nn.Linear(dff, d_model)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class ETransformer(nn.Module):
    def __init__(self, d_model, num_heads, dff):
        super(ETransformer, self).__init__()
        self.mha = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, dff)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        att_output = self.mha(x)
        out1 = self.layernorm1(x + att_output)
        ffn_output = self.ffn(out1)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2


class LeFF(nn.Module):
    def __init__(self, d_model, dff):
        super(LeFF, self).__init__()
        self.depthwise_conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, dff, kernel_size=1)
        self.depthwise_conv2 = nn.Conv1d(dff, dff, kernel_size=3, padding=1, groups=dff)
        self.pointwise_conv2 = nn.Conv1d(dff, d_model, kernel_size=1)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x):
        out = self.depthwise_conv1(x.permute(0, 2, 1))
        out = torch.nn.functional.relu(self.pointwise_conv1(out))
        out = self.depthwise_conv2(out)
        out = torch.nn.functional.relu(self.pointwise_conv2(out))
        out = self.layernorm(out.permute(0, 2, 1) + x)
        return out

class SformerBlock(nn.Module):
    def __init__(self, d_model, M, num_heads, dff):
        super(SformerBlock, self).__init__()
        self.etransformers = nn.ModuleList([ETransformer(d_model, num_heads, dff) for _ in range(M)])
        self.leff = LeFF(d_model, dff)

    def forward(self, x):
        for etransformer in self.etransformers:
            x = etransformer(x)
        x = self.leff(x)
        return x

class DenSformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, M):
        super(DenSformer, self).__init__()
        self.sformer_blocks = nn.ModuleList([SformerBlock(d_model, M, num_heads, dff) for _ in range(num_layers)])
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x):
        for sformer_block in self.sformer_blocks:
            x = sformer_block(x)
        x = self.layernorm(x)
        return x

# Example usage
num_layers = 6
d_model = 512
num_heads = 8
dff = 2048
M = 4

densformer = DenSformer(num_layers, d_model, num_heads, dff, M)
input_data = torch.randn(32, 8,512, 32)  # Example input data
output_data = densformer(input_data)  # Forward pass through the DenSformer
