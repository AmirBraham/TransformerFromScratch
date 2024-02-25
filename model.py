import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.d_model = d_model # the size of the vector Embedding
        self.vocab_size = vocab_size # Vocabulary size
        # An embedding layer simple lookup table that maps an index value to a weight matrix of a certain dimension.
        self.embedding = nn.Embedding(vocab_size,d_model)
    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self,d_model:int,seq_len:int,dropout:float):
        super().__init__()
        #d_model is the size of the vector
        self.d_model = d_model
        #seq_len is the length of the sentence
        
        self.seq_len = seq_len
        #Adding dropout to avoid overfitting
        self.dropout = nn.Dropout(dropout)

        # Matrix of shape(seq_len,d_model)
        positional_encoding_matrix = torch.zeros(self.seq_len,self.d_model)
        #Creating a tensor of shape (seq_len,1)
        pos = torch.arange(0, self.seq_len).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0,self.d_model,2).float() * (-math.log(10000)/ d_model))
        
        
        positional_encoding_matrix[:,0::2] = torch.sin(pos*div_term)
        positional_encoding_matrix[:,1::2] = torch.cos(pos*div_term)
        
        # Adding batch dimension to apply the positional_encoding_matrix to a batch of sentences
        positional_encoding_matrix = positional_encoding_matrix.unsqueeze(0) # the positional encoding is of shape (1,seq_len,d_model)
        
        # Register positional_encoding_matrix to buffer (saved with the model , not a learnable param)
        self.register_buffer('positional_encoding_matrix',positional_encoding_matrix)

    def forward(self,x):
        x = x+(self.positionalEncodingMatrix[:,x.shape[1],:]).requires_grad(False)
        return self.dropout(x)



class LayerNormalization(nn.Module):
    def __init__(self,eps:float = 1e-5):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
    def forward(self,x):
        mean_x = x.mean(dim=-1,keepdim=True) # Calculate mean of last dimension
        std_x = x.std(dim=-1,keepdim=True)
        return self.alpha * (x-mean_x)/(std_x+self.eps) + self.bias
        

class FeedForward(nn.Module):
    def __init__(self,d_model:int,d_ff:int,dropout:float):
        super().__init__()
        self.dropout = nn.Dropout(dropout) 
        self.linear_1=nn.Linear(d_model,d_ff)#W1 and B1
        self.linear_2=nn.Linear(d_ff,d_model) # W2 and B2
    def forward(self,x):
        # the input is of type : (Batch,Seq_len,d_model)
        #(Batch,Seq_len,d_model) -> (Batch,Seq_len,d_ff) -> (Batch,Seq_len,d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
        
        
        