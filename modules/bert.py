import torch
from torch import nn
from torch.nn import functional as F
from modules.layer import *

#------------------------------------#

# This part tends to realize BertModel、Bert-pretraining base on layer.py


# 1、Here we first set the interface that initial our model's weight

def init__weight(module):
    if isinstance(module,nn.Linear):#This is for Linear 
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module,nn.Embedding):
        module.weight.data.normal_(mean=0.0,std = 0.02)
        if module.padding_idx is not None:#这里初始化的是padding_idx
            module.weight.data[module.padding_idx].zero_()

    elif isinstance(module,nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


#  2、Here we create out bert model
class BertPooler(nn.Module):
    def __init__(self,hidden_size):
        '''
        This is the pooling layer for BERT, which takes the first token's hidden state (CLS token) and applies a linear transformation followed by a tanh activation.
        '''
        super().__init__()
        self.hidden_size = hidden_size
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self,hidden_states:torch.Tensor) ->torch.Tensor:
        '''
        param hidden_states: hidden states from the last layer of BERT, shape [batch_size, seq_len, hidden_size]
        return: pooled output, shape [batch_size, hidden_size]
        '''
        hidden_states = hidden_states[:, 0]
        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertModule(nn.Module):
    '''
    As we define in the layer.py,the bert model consist of:
        1、Embedding
        2、Transformer Encoder
        3、Pooling Layer (optional)
    So we need surpass each parameter
    '''
    def __init__(self,vocab_size,type_vocab_size,hidden_size,max_len,num_layers,intermidiate_size,num_attention_head,dropout,add_pooling_layer = True,padding_idx = None):
        '''
        param vocab_size: vocabulary size
        param type_vocab_size: token type vocabulary size
        param hidden_size: embedding dimension
        param max_len: maximum sequence length
        param num_layers: number of transformer encoder layers
        param intermidiate_size: intermediate size for feed-forward network
        param num_attention_head: number of attention heads
        param dropout: dropout rate
        param add_pooling_layer: whether to add a pooling layer for sequence classification
        param padding_idx: padding index for embedding layer
        '''
        super().__init__()
        #Here we first create the embedding layer
        self.embeddings = BertEmbeddings(
            vocab_size=vocab_size,
            type_vocab_size=type_vocab_size,
            hidden_size=hidden_size,
            max_len=max_len,
            dropout=dropout,
            pad_token_idx=padding_idx
        )
        #Then we create the Bert Encoder
        self.encoder = BertEncoder(
            hidden_size=hidden_size,
            num_layers=num_layers,
            intermidiate_size=intermidiate_size,
            num_attention_head=num_attention_head,
            dropout=dropout
        )
        self.BertPooler = BertPooler(hidden_size = hidden_size) if add_pooling_layer else None

        self.apply(init__weight)  # Initialize weights

    def forward(self,input_ids:torch.LongTensor,token_type_idx:Optional[torch.LongTensor] = None,attention_mask:torch.Tensor = None) ->torch.Tensor:
        '''
        param input_ids: input token ids, shape [batch_size, seq_len]
        param token_type_idx: input token type ids, shape [batch_size, seq_len], optional
        param attention_mask: attention mask, shape [batch_size, seq_len], optional
        return: encoded output, shape [batch_size, seq_len, hidden_size]
        '''
        if attention_mask is not None and attention_mask.dim() == 2:
        # [batch_size, seq_len] -> [batch_size, 1, seq_len, seq_len]
            attention_mask = attention_mask.unsqueeze(1).repeat(1, input_ids.size(1), 1).unsqueeze(1)
 
        #First Embeddings
        embeddings = self.embeddings(input_ids,token_type_idx)
        #Then Encoder
        Encoder_output = self.encoder(embeddings,attention_mask)
        #Now because we have the pooling layer，we can get the pooled output
        pooled_output = self.BertPooler(Encoder_output) if self.BertPooler is not None else None
        return Encoder_output,pooled_output
        


# 3.Here we create the BertPretrainingModel, which is used for pre-training tasks like masked language modeling and next sentence prediction

# Because the mask-procedure has been done in the dataset, so we just need to create the BertMLM and BertNSP layers

class BertPrediction(nn.Module):
    def __init__(self,hidden_size,vocab_size):
        '''
        This is the prediction layer for BERT, which predicts the masked tokens.
        '''
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.decoder = nn.Linear(hidden_size, vocab_size)
    
    def forward(self,hidden_states:torch.Tensor)->torch.Tensor:
        '''
        param hidden_states: hidden states from the last layer of BERT, shape [batch_size, seq_len, hidden_size]
        return: predicted token logits, shape [batch_size, seq_len, vocab_size]
        '''
        hidden_states = self.dense(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        prediction = self.decoder(hidden_states)
        return prediction

class BertMLM(nn.Module):
    def __init__(self,hidden_size,vocab_size):
        '''
        This is the masked language modeling layer for BERT.
        '''
        super().__init__()
        self.prediction = BertPrediction(hidden_size, vocab_size)

    def forward(self,hidden_states:torch.Tensor)->torch.Tensor:
        '''
        param hidden_states: hidden states from the last layer of BERT, shape [batch_size, seq_len, hidden_size]
        return: predicted token logits, shape [batch_size, seq_len, vocab_size]
        '''
        return self.prediction(hidden_states)
    
class BertNSP(nn.Module):
    def __init__(self,hidden_size):
        '''
        This is the next sentence prediction layer for BERT'''
        super().__init__()
        self.seq_relationship = nn.Linear(hidden_size,2)

    def forward(self,pooled_output:torch.Tensor)->torch.Tensor:
        '''
        param pooled_output: pooled output from the last layer of BERT, shape [batch_size, hidden_size]
        return: next sentence prediction logits, shape [batch_size, 2]
        '''
        return self.seq_relationship(pooled_output)
    
        
# 4.After we have created the BertMLM and BertNSP, we can create the BertPretrainingModel
class BertPretrainingModel(nn.Module):
    def __init__(self,vocab_size,type_vocab_size,hidden_size,max_len,num_layers,intermidiate_size,num_attention_head,dropout,padding_idx = None):
        '''
        param vocab_size: vocabulary size
        param type_vocab_size: token type vocabulary size
        param hidden_size: embedding dimension
        param max_len: maximum sequence length
        param num_layers: number of transformer encoder layers
        param intermidiate_size: intermediate size for feed-forward network
        param num_attention_head: number of attention heads
        param dropout: dropout rate
        param padding_idx: padding index for embedding layer
        '''
        super().__init__()
        self.bert = BertModule(
            vocab_size=vocab_size,
            type_vocab_size=type_vocab_size,
            hidden_size=hidden_size,
            max_len=max_len,
            num_layers=num_layers,
            intermidiate_size=intermidiate_size,
            num_attention_head=num_attention_head,
            dropout=dropout,
            add_pooling_layer=True,
            padding_idx=padding_idx
        )
        self.vocab_size = vocab_size
        self.mlm = BertMLM(hidden_size,vocab_size)
        self.nsp = BertNSP(hidden_size)

    def forward(self,
                input_ids:torch.LongTensor,token_type_idx:Optional[torch.LongTensor] = None,
                attention_mask:Optional[torch.Tensor] = None,labels = None, next_sentence_label = None):
        '''
        param input_ids: input token ids, shape [batch_size, seq_len]
        param token_type_idx: input token type ids, shape [batch_size, seq_len], optional
        param attention_mask: attention mask, shape [batch_size, seq_len], optional
        param labels: labels for masked language modeling, shape [batch_size, seq_len], optional
        param next_sentence_label: labels for next sentence prediction, shape [batch_size], optional
        return: output logits and loss if labels are provided
        '''
        sequence_output, pooled_output = self.bert(input_ids, token_type_idx, attention_mask)
        mlm_logits = self.mlm(sequence_output)
        nsp_logits = self.nsp(pooled_output)
        outputs = (mlm_logits, nsp_logits)
        if labels is not None:
            # Calculate masked language modeling loss
            mlm_loss = F.cross_entropy(mlm_logits.view(-1, self.vocab_size), labels.view(-1), ignore_index=-1)
            outputs = (mlm_loss,) + outputs
        if next_sentence_label is not None:
            # Calculate next sentence prediction loss
            nsp_loss = F.cross_entropy(nsp_logits, next_sentence_label)
            outputs = (nsp_loss,) + outputs
        return outputs
