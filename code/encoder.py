import math
import torch
from torch import nn

class MultiheadAttention(nn.Module):
    def __init__(self,d_model,num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model//num_heads
        
        self.wq = nn.Linear(d_model,d_model)
        self.wk = nn.Linear(d_model,d_model)
        self.wv = nn.Linear(d_model,d_model)

        self.fc_out = nn.Linear(d_model,d_model)

    def split_heads(self,x):
        B = x.size(0)
        x = x.view(B,-1,self.num_heads,self.depth)
        return x
    def forward(self,q,k,v):
        # consider the sentance the "cat sat on the mat"
        # q represents what "sat" wants to focus on
        # k is for all words — "The", "cat", "sat", "on", "the", "mat" — representing what each word offers
        # v is the value of each word — like the meaning or context of "cat", "on", etc.
        # split into multiple heads so "sat" can look at different aspects (syntax, meaning, etc.)
        # scores is how much "sat" relates to each word (e.g., "cat": 2.1, "on": 1.7, ...)
        # attention_weights is softmax of scores — how much "sat" pays attention to each word (like "cat": 0.4)
        # output is the new vector for "sat", built from the weighted sum of values ("cat", "on", etc.)
        # bring all the attention heads back together

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        #what makes this multihead
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)


        scores = torch.matmul(q,k.transpose(-2,-1))/math.sqrt(self.depth)
        attention_weights = torch.softmax(scores,-1)
        output = torch.matmul(attention_weights,v)
        
        output = output.transpose(1,2).contiguous()

        

        output = output.view(q.size(0),-1,self.d_model)

        return self.fc_out(output)

class PositionalFeedForwardNetwork(nn.Module):
    def __init__(self,d_model,dff):
        super().__init__()

        self.f1 = nn.Linear(d_model,dff)
        self.f2 = nn.Linear(dff,d_model)

    def forward(self,x):
        # it is positional since linear works independantly for each i in the sequence
        x = self.f1(x)
        x = torch.relu(x)
        x = self.f2(x)
        return x