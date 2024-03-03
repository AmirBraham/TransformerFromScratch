import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model  # the size of the vector Embedding
        self.vocab_size = vocab_size  # Vocabulary size
        # An embedding layer simple lookup table that maps an index value to a weight matrix of a certain dimension.
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        # d_model is the size of the vector
        self.d_model = d_model
        # seq_len is the length of the sentence

        self.seq_len = seq_len
        # Adding dropout to avoid overfitting
        self.dropout = nn.Dropout(dropout)

        # Matrix of shape(seq_len,d_model)
        positional_encoding_matrix = torch.zeros(self.seq_len, self.d_model)
        # Creating a tensor of shape (seq_len,1)
        pos = torch.arange(0, self.seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float() * (-math.log(10000) / d_model)
        )

        positional_encoding_matrix[:, 0::2] = torch.sin(pos * div_term)
        positional_encoding_matrix[:, 1::2] = torch.cos(pos * div_term)

        # Adding batch dimension to apply the positional_encoding_matrix to a batch of sentences
        positional_encoding_matrix = positional_encoding_matrix.unsqueeze(
            0
        )  # the positional encoding is of shape (1,seq_len,d_model)

        # Register positional_encoding_matrix to buffer (saved with the model , not a learnable param)
        self.register_buffer("positional_encoding_matrix", positional_encoding_matrix)

    def forward(self, x):
        x = x + (self.positionalEncodingMatrix[:, x.shape[1], :]).requires_grad(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean_x = x.mean(dim=-1, keepdim=True)  # Calculate mean of last dimension
        std_x = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean_x) / (std_x + self.eps) + self.bias


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(d_model, d_ff)  # W1 and B1
        self.linear_2 = nn.Linear(d_ff, d_model)  # W2 and B2

    def forward(self, x):
        # the input is of type : (Batch,Seq_len,d_model)
        # (Batch,Seq_len,d_model) -> (Batch,Seq_len,d_ff) -> (Batch,Seq_len,d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h  # Number of heads
        # d_model should be divisible by h
        assert d_model % h == 0, "d_model should be divisible by h"
        self.d_k = d_model // h
        # init Wq,Wk,Wv : learnable matrices to multiply by  Q,K,V 
        self.w_q = nn.Linear(d_model, d_model) 
        self.w_k = nn.Linear(d_model, d_model) 
        self.w_v = nn.Linear(d_model, d_model)
        # In the paper , d_k = d_v
        self.w_o = nn.Linear(d_model,d_model)
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query,key,value,mask,dropout:nn.Dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k) # we transpose the last 2 dimensions of Key
        # Attention scores is of shape (Batch,h,seq_len,seq_len)
        if mask is not None:
            attention_scores.masked_fill_(mask==0,-1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value),attention_scores
    
    def forward(self,q,k,v,mask):
        query = self.w_q(q) # (Batch,seq_len,d_model) ->(Batch,seq_len,d_model)
        key = self.w_k(k) # (Batch,seq_len,d_model) ->(Batch,seq_len,d_model)
        value = self.w_v(v) # (Batch,seq_len,d_model) ->(Batch,seq_len,d_model)
        
        # (Batch,seq_len,d_model) -> (Batch,seq_len,h,d_k)  -> (Batch,h,seq_len,d_k)
        query = query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)
        key = key.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(1,2)
        value = key.view(value.shape[0],value.shape[1],self.h,self.d_k).transpose(1,2)
        
        x , self.attention_scores = MultiHeadAttention.attention(query,key,value,mask,self.dropout)
        #(Batch,h,seq_len,d_k) -> (Batch,seq_len,h,_dk) -> (Batch,seq_len,d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h*self.d_k)
        #(Batch,seq_len,d_model) -> (Batch,seq_len,d_model)
        return self.w_o(x)
    
class ResidualConnection(nn.Module):
    def __init__(self,dropout:float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
    def forward(self,x,sublayer):
        #sublayer is the previous layer
        return x + self.dropout(sublayer(self.norm(x)))
        
class EncoderBlock(nn.Module):
    def __init__(self,self_attention:MultiHeadAttention,feed_forward:FeedForward,dropout:float) -> None:
        super().__init__()
        self.self_attention= self_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout),ResidualConnection(dropout)])
    def forward(self,x,src_mask):
        x = self.residual_connections[0](x,lambda x:self.self_attention(x,x,x,src_mask))
        x = self.residual_connections[1](x,self.feed_forward)
        return x 
    
class Encoder(nn.Module):
    def __init__(self,layers:nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    def forward(self,x,mask):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)
            