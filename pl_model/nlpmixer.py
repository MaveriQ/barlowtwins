import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from transformers import BertModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class MixerBlock(nn.Module):

    def __init__(self, embed_dim, num_tokens, token_hid_dim, embedding_hid_dim, dropout = 0.):
        super().__init__()

        self.token_mix = nn.Sequential(
            nn.LayerNorm(embed_dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_tokens, token_hid_dim, dropout),
            Rearrange('b d n -> b n d')
        )

        self.embedding_mix = nn.Sequential(
            nn.LayerNorm(embed_dim),
            FeedForward(embed_dim, embedding_hid_dim, dropout),
        )

    def forward(self, x,attention_mask):
#         pdb.set_trace()
        att_exp = attention_mask.unsqueeze(-1).expand(x.size()).float()
        x = torch.mul(x,att_exp)
        
        x = x + self.token_mix(x)

        x = x + self.embedding_mix(x)

        return x

class NLPMixer(nn.Module):

    def __init__(self, config, 
                num_layers=None, 
                num_tokens=None, 
                embed_dim=None, 
                token_hid_dim=None, 
                embedding_hid_dim=None, 
                do_mlm=False, 
                do_embed=True,
                all_hidden_states=False):
        super().__init__()

        self.do_mlm = do_mlm
        self.do_embed = do_embed

        if num_tokens is None:
            num_tokens=config.max_position_embeddings
        if embed_dim is None:
            embed_dim = config.hidden_size
        if embedding_hid_dim is None:
            embedding_hid_dim = config.intermediate_size
        if token_hid_dim is None:
            token_hid_dim = config.intermediate_size
        if num_layers is None:
            num_layers = config.num_hidden_layers

        self.all_hidden_states = config.output_hidden_states
        if self.do_embed:
            self.embedding = BertModel(config).from_pretrained('bert-base-uncased').embeddings

        self.mixer_blocks = nn.ModuleList([])

        for _ in range(num_layers):
            self.mixer_blocks.append(MixerBlock(embed_dim, num_tokens, token_hid_dim, embedding_hid_dim))

        self.layer_norm = nn.LayerNorm(embed_dim)

        if self.do_mlm:
            self.bert_mlm_head = BertOnlyMLMHead(config)

    def forward(self,
                attention_mask,
                input_ids=None,
                token_type_ids=None,
                position_ids=None,
                bert_output=None,
                ):


        if self.do_embed:
            x = self.embedding(input_ids=input_ids,
                                token_type_ids=token_type_ids,
                                position_ids=position_ids)
        else:
            if self.all_hidden_states:
                x = torch.mean(torch.stack(bert_output.hidden_states),0)
            else:
                x = bert_output.last_hidden_state
        
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x,attention_mask)

        x = self.layer_norm(x)

        # x = x.mean(dim=1) #implemented it in the pooler later

        if self.do_mlm:
            return {'token_embeddings':x,
                    'logits':self.bert_mlm_head(x)}
        else:
            return {'token_embeddings':x}