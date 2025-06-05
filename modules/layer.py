import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional
#-------------------------------------------#

# task: This part mainly implements the BERT-Encoder, without the output head

#-------------------------------------------#

#------------------------------------#

#       1. First, complete the Embedding          #

'''
Convert input_ids to embedding input:
1. Word embedding - word_embeddings
2. Position embedding - position-embeddings
3. Token type embedding - token-type-embeddings
4. Finally, pass through Layer-Norm and Dropout
The encoded output shape: [batch_size, seq_len, hidden_size]
'''

class BertEmbeddings(nn.Module):
    def __init__(self,vocab_size,type_vocab_size,hidden_size,max_len,dropout = 0.1,pad_token_idx = None):   
        '''
        param vocab_size: vocabulary size
        param type_vocab_size: token type vocabulary size
        param hidden_size: embedding dimension
        param max_len: maximum sequence length
        param dropout: dropout rate
        param pad_token_idx: padding token id
        This is the BERT Embedding layer, including word, position, and token type embeddings
        '''
        super(BertEmbeddings,self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size,hidden_size,padding_idx=pad_token_idx) # word embedding layer, input dim vocab_size, output hidden_size
        self.position_embeddings = nn.Embedding(max_len,hidden_size) # position embedding layer, input max_len, output hidden_size
        self.token_type_embeddings = nn.Embedding(type_vocab_size,hidden_size) # token type embedding layer, input type_vocab_size, output hidden_size
        # All embeddings are projected to hidden_size for easy addition

        self.layernorm = nn.LayerNorm(hidden_size) # normalization layer to accelerate training
        self.dropout = nn.Dropout(dropout) # regularization layer

    def forward(self,input_ids:torch.LongTensor,token_type_idx:Optional[torch.LongTensor] = None) ->torch.Tensor:
        '''
        param input_ids: input token ids, shape [batch_size, seq_len]
        param token_type_idx: input token type ids, shape [batch_size, seq_len], optional
        return: encoded output, shape [batch_size, seq_len, hidden_size]
        '''
        batch_size,seq_len = input_ids.size()
        
        # If token_type_idx is None, pad with zeros for addition
        if token_type_idx is None:
            token_type_idx = torch.zeros_like(input_ids,dtype=torch.long)

        # Generate position indices
        position_info = torch.arange(seq_len,dtype = torch.long,device=input_ids.device).expand(1,-1)

        # Pass through embedding layers
        w_embeddings = self.word_embeddings(input_ids)
        p_embeddings = self.position_embeddings(position_info)
        t_embeddings = self.token_type_embeddings(token_type_idx)
        embeddings = w_embeddings + p_embeddings + t_embeddings
        embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    

#-------------------------------------------------------#

#      Above we completed the encoding for the Encoder structure           #

#   2. Next, build the Transformer Encoder part           #

'''
Recall the Transformer structure, after input:
1. Multi-head self-attention: Self-MultiHeadAttention
2. Residual & NormLayer
3. FFN

'''

#---------------------------------------------------------#

#       2.1 Next is to build the multi-head self-attention mechanism                    #

'''
Refer to the formula in the diagram.
Since Encoder does not require masking, implementation is straightforward.
'''

class SelfAttention(nn.Module):
    def __init__(self,hidden_size,num_heads,dropout):
        '''
        param hidden_size: embedding dimension
        param num_heads: number of attention heads
        param dropout: dropout rate
        '''
        super().__init__()
        self.num_heads = num_heads # number of heads, splits feature dimension for parallel attention
        self.attention_size = hidden_size // num_heads # dimension per attention head
        self.all_head_size = hidden_size # total feature dimension

        # Define Q, K, V
        '''
        Q = X · Wq
        K = X · Wk
        V = X · Wv
        '''
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        # Use torch's scaled_dot_product_attention if available
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,mask = None):
        '''
        param x: input encoding, shape [batch_size, seq_len, hidden_size]
        param mask: optional mask, shape [batch_size, seq_len, seq_len]
        return: output after attention, shape [batch_size, seq_len, hidden_size]
        '''
        batch_size,num_head,attention_head = x.size(0),self.num_heads,self.attention_size

        # Compute Q, K, V
        q = self.query(x).view(batch_size,-1,num_head,attention_head).transpose(1,2)
        '''
            Input [bsz, seq_len, hidden_size] → Output [bsz, seq_len, all_head_size]
            all_head_size = nh * nd

            .view(bsz, seq_len, num_head, attention_head)
            Reshape to: [bsz, seq_len, num_heads, head_dim]

            .transpose(1, 2)
            Change to: [bsz, num_heads, seq_len, head_dim] for parallel attention computation.
        '''
        k = self.key(x).view(batch_size,-1,num_head,attention_head).transpose(1, 2)
        v = self.value(x).view(batch_size,-1,num_head,attention_head).transpose(1, 2)

        if self.flash:
            att = F.scaled_dot_product_attention(q,k,v,dropout_p=self.dropout.p if self.training else 0.0,is_causal=False,attn_mask=mask,scale=None)

        y = att.transpose(1, 2).contiguous().view(batch_size, -1, self.all_head_size) # reshape back to [batch_size, seq_len, hidden_size]

        # output projection will be performed later
        return y
    

#------------------------------------------------#

#   2.2 Next, build a single BERT_layer               #

'''
Similar to the Transformer architecture, you need to build a single layer first.
A BERT_layer includes: a self-attention layer, residual & norm, an FNN, then another residual & norm.
'''
class Bertlayer(nn.Module):
    def __init__(self,hidden_size,intermidiate_size,num_attention_heads,dropout):
        '''
        param hidden_size: embedding dimension
        param intermidiate_size: intermediate layer dimension
        param num_attention_heads: number of attention heads
        param dropout: dropout rate
        This is the BERT Encoder layer, including self-attention, residual connection, and feed-forward network
        '''
        super(Bertlayer, self).__init__()
        # Self-attention layer
        self.attention = SelfAttention(hidden_size,num_attention_heads,dropout)
        # LayerNorm after attention
        self.attention_layernorm = nn.LayerNorm(hidden_size)

        # FFN: Feed-forward neural network
        self.FFn_input = nn.Linear(hidden_size,intermidiate_size)
        self.FFn_output = nn.Linear(intermidiate_size,hidden_size)

        # LayerNorm after FFN
        self.FFnlayernorm = nn.LayerNorm(hidden_size)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,mask = None):
        '''
        param x: input encoding, shape [batch_size, seq_len, hidden_size]
        param mask: optional mask, shape [batch_size, seq_len, seq_len]
        return: output encoding, shape [batch_size, seq_len, hidden_size]
        '''
        # overall Framework: SelfAttention -> Dropout -> LayerNorm&add -> FFn -> Dropout -> LayerNorm&add
        # Self-attention
        attention_output = self.attention(x,mask)
        # Dropout
        attention_output = self.dropout(attention_output)
        # LayerNorm after attention
        attention_output = self.attention_layernorm(attention_output + x)

        # Feed-forward network
        intermidiate_output = F.gelu(self.FFn_input(attention_output))
        ffn_output = self.FFn_output(intermidiate_output)
        # Dropout
        ffn_output = self.dropout(ffn_output)
        # LayerNorm after FFN
        ffn_output = self.FFnlayernorm(ffn_output + attention_output)
        return ffn_output
    



#------------------------------------------------#

#   2.3 Next, build the BERT_Encoder                  #

#------------------------------------------------#

'''
Similar to the Transformer, after building a single layer, stack multiple layers.
'''

class BertEncoder(nn.Module):
    def __init__(self,hidden_size,num_layers,intermidiate_size,num_attention_head,dropout):
        '''
        param hidden_size: embedding dimension
        param num_layers: number of Encoder layers
        param intermidiate_size: intermediate layer dimension
        param num_attention_heads: number of attention heads
        param dropout: dropout rate
        This is the BERT Encoder, containing multiple BERT layers
        '''
        super(BertEncoder, self).__init__()
        self.layers = nn.ModuleList([Bertlayer(hidden_size,intermidiate_size,num_attention_head,dropout) for _ in range(num_layers)])

    def forward(self,x,mask = None):
        '''
        param x: input encoding, shape [batch_size, seq_len, hidden_size]
        param mask: optional mask, shape [batch_size, seq_len, seq_len]
        return: output encoding, shape [batch_size, seq_len, hidden_size]
        '''
        for layer in self.layers:
            x = layer(x,mask)
        return x








