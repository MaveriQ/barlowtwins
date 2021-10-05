from collections import namedtuple
import os, sys
import json
from torch import optim
from transformers import BertModel, BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertLMPredictionHead
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
import math
import torch.nn.functional as F
import pdb
from datasets import load_from_disk
from transformers import DataCollatorForLanguageModeling, BertTokenizerFast, CONFIG_MAPPING
from nlpmixer import NLPMixer
from barlowbert_dm import DataCollatorForBarlowBertWithMLM

ACT2FN = {
    "relu": F.relu,
    "silu": F.silu,
    "swish": F.silu,
    "gelu": F.gelu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
}

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

def CosineSimilarity(x1,x2):
    x1n = x1/x1.norm(dim=1, keepdim=True)
    x2n = x2/x2.norm(dim=1, keepdim=True)

    return x1n@x2n.transpose(0,1)

def get_diag_loss(loss_dict,c,weight,typ):
    on_diag = torch.diagonal(c).add_(-1).pow_(2).mean()
    off_diag = off_diagonal(c).pow_(2).mean()
    loss_dict[f'{typ}_ondiag'] = on_diag
    loss_dict[f'{typ}_offdiag'] = off_diag
    return off_diag + weight * on_diag

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class Projector(nn.Module):
    def __init__(self, config,input_size,output_size):
        super().__init__()
        self.dense = nn.Linear(input_size, output_size,bias=False)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(output_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class Pooler(torch.nn.Module):
    """Performs pooling (max or mean) on the token embeddings.
    Using pooling, it generates from a variable sized sentence a fixed sized sentence embedding. This layer also allows to use the CLS token if it is returned by the underlying word embedding model.
    You can concatenate multiple poolings together.
    :param word_embedding_dimension: Dimensions for the word embeddings
    :param pooling_mode: Can be a string: mean/max/cls. If set, overwrites the other pooling_mode_* settings
    :param pooling_mode_cls_token: Use the first token (CLS token) as text representations
    :param pooling_mode_max_tokens: Use max in each dimension over all tokens.
    :param pooling_mode_mean_tokens: Perform mean-pooling
    :param pooling_mode_mean_sqrt_len_tokens: Perform mean-pooling, but devide by sqrt(input_length).
    """
    def __init__(self,
                 word_embedding_dimension: int,
                 all_hidden_states: bool = True,
                 pooling_mode: str = None,
                 pooling_mode_cls_token: bool = False,
                 pooling_mode_max_tokens: bool = False,
                 pooling_mode_mean_tokens: bool = True,
                 pooling_mode_mean_sqrt_len_tokens: bool = False,
                 mean_cat: str = 'cat'
                 ):
        super(Pooler, self).__init__()

        self.config_keys = ['word_embedding_dimension',  'pooling_mode_cls_token', 'pooling_mode_mean_tokens', 'pooling_mode_max_tokens', 'pooling_mode_mean_sqrt_len_tokens']

        if pooling_mode is not None:        #Set pooling mode by string
            pooling_mode = pooling_mode.lower()
            assert pooling_mode in ['mean', 'max', 'cls']
            pooling_mode_cls_token = (pooling_mode == 'cls')
            pooling_mode_max_tokens = (pooling_mode == 'max')
            pooling_mode_mean_tokens = (pooling_mode == 'mean')

        self.all_hidden_states = all_hidden_states
        self.word_embedding_dimension = word_embedding_dimension
        self.pooling_mode_cls_token = pooling_mode_cls_token
        self.pooling_mode_mean_tokens = pooling_mode_mean_tokens
        self.pooling_mode_max_tokens = pooling_mode_max_tokens
        self.pooling_mode_mean_sqrt_len_tokens = pooling_mode_mean_sqrt_len_tokens
        self.pool_method = mean_cat

        pooling_mode_multiplier = sum([pooling_mode_cls_token, pooling_mode_max_tokens, pooling_mode_mean_tokens, pooling_mode_mean_sqrt_len_tokens])
        if self.pool_method=='cat':
            self.pooling_output_dimension = (pooling_mode_multiplier * word_embedding_dimension)
        elif self.pool_method=='mean':
            self.pooling_output_dimension = word_embedding_dimension


    def __repr__(self):
        return "Pooling({})".format(self.get_config_dict())

    def get_pooling_mode_str(self) -> str:
        """
        Returns the pooling mode as string
        """
        modes = []
        if self.pooling_mode_cls_token:
            modes.append('cls')
        if self.pooling_mode_mean_tokens:
            modes.append('mean')
        if self.pooling_mode_max_tokens:
            modes.append('max')
        if self.pooling_mode_mean_sqrt_len_tokens:
            modes.append('mean_sqrt_len_tokens')

        return "+".join(modes)

    def forward(self, attention_mask,
                    bert_output=None,
                    nlpmixer_output=None,
                    ):

        if bert_output is not None:
            if self.all_hidden_states:
                token_embeddings = torch.mean(torch.stack(bert_output.hidden_states),0)
            else:
                token_embeddings = bert_output.last_hidden_state
        elif nlpmixer_output is not None:
            token_embeddings = nlpmixer_output['token_embeddings']
        else:
            print('Error: Either of bert_output or nlpmixer_output must be set')
            sys.exit(1)            

        # pdb.set_trace()
        ## Pooling strategy
        output_vectors = []
        if self.pooling_mode_cls_token:
            cls_token = token_embeddings[:,0]
            output_vectors.append(cls_token)
        if self.pooling_mode_max_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            max_over_time = torch.max(token_embeddings, 1)[0]
            output_vectors.append(max_over_time)
        if self.pooling_mode_mean_tokens or self.pooling_mode_mean_sqrt_len_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)

            if self.pooling_mode_mean_tokens:
                output_vectors.append(sum_embeddings / sum_mask)
            if self.pooling_mode_mean_sqrt_len_tokens:
                output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))

        if self.pool_method=='cat':
            output_vector = torch.cat(output_vectors, 1)
        elif self.pool_method=='mean':
            output_vector = torch.stack(output_vectors,dim=2).mean(dim=2)
            # pdb.set_trace()
        return {'sentence_embedding':output_vector,
                'token_embeddings':token_embeddings}

    def get_sentence_embedding_dimension(self):
        return self.pooling_output_dimension

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path):
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)

        return Pooler(**config)

class SentenceBertWithPooler(BertPreTrainedModel):
    
    def _init_weights(self, module):
        """Initialize the weights"""
        # pdb.set_trace()
        if isinstance(module,Pooler) or isinstance(module,Projector): # init weights only for non Bert modules as we are using pretrained weights for rest.
            if isinstance(module, nn.Linear):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            
    def __init__(self, config,args):
        super().__init__(config)

        self.config = config
        self.args = args
            
        self.bert = BertModel(self.config, add_pooling_layer=False).from_pretrained('bert-base-uncased')

        if self.args.do_mlm:
            self.lm_head = BertLMPredictionHead(config)           
            for params in self.bert.pooler.parameters():
                params.requires_grad=False
        else:
            for params in self.bert.parameters():
                params.requires_grad=False            

        self.pooler = Pooler(word_embedding_dimension=self.config.hidden_size,
                            pooling_mode_cls_token=False,#,self.args.cls_pooling, #TODO Fix it
                            pooling_mode_max_tokens=False,#self.args.max_pooling,
                            pooling_mode_mean_tokens=True,#self.args.mean_pooling,
                            mean_cat='cat',
                            all_hidden_states=self.config.output_hidden_states)#self.args.pool_type)

        sizes = [self.pooler.get_sentence_embedding_dimension()] + list(map(int, self.args.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(Projector(config,sizes[i], sizes[i + 1]))
        layers.append(Projector(config,sizes[-2], sizes[-1]))
        self.projector = nn.Sequential(*layers)
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)
        
        self.apply(self._init_weights)
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=self.config.output_hidden_states,
            return_dict=return_dict,
        )

        if self.args.do_mlm:
            mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
            mlm_outputs = self.bert(
                mlm_input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=False, #True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
                return_dict=True,
            )
            prediction_scores = self.lm_head(mlm_outputs.last_hidden_state)
        # pdb.set_trace()
        pooled = self.pooler(bert_output = outputs,attention_mask=attention_mask)
        
        projection = self.projector(pooled)

        if self.args.do_mlm:
            return {'sentence_embedding':projection,
                    'prediction_scores':prediction_scores}
        else:        
            return {'sentence_embedding':projection}

class SentenceBertWithNLPMixer(BertPreTrainedModel):
    
    def _init_weights(self, module):
        """Initialize the weights"""
        # pdb.set_trace()
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _process_bert_encoder(self): #Processing BERT to make only last n layers trainable.

        self.bert.train() # by default a pretrained model is in eval mode but requires_grad is set to True

        for layer in self.bert.encoder.layer[:self.bert.config.num_hidden_layers-self.args.num_trainable_layers]: #modifying layers other than last n
            layer.eval()
            for param in layer.parameters():
                param.requires_grad=False

        for params in self.bert.pooler.parameters():
            params.requires_grad=False

    def __init__(self, config,args=None):
        super().__init__(config)

        self.config = config
        self.args = args
  
        # if self.args.skip_bert:
        #     self.nlpmixer = NLPMixer(self.config,num_layers=self.args.num_mixer_layers,do_embed=True)
        # else:

        if self.args.mlm_weight > 0:
            self.lm_head = BertLMPredictionHead(config)
            assert self.num_trainable_layers > 0, "number of trainable layers should be > 0 for mlm"           
        
        if self.args.num_mixer_layers > 0:
            self.nlpmixer = NLPMixer(self.config,num_layers=self.args.num_mixer_layers,do_embed=False)

        self.pooler = Pooler(word_embedding_dimension=self.config.hidden_size,
                            pooling_mode_cls_token=False,#,self.args.cls_pooling, #TODO Fix it
                            pooling_mode_max_tokens=False,#self.args.max_pooling,
                            pooling_mode_mean_tokens=True,#self.args.mean_pooling,
                            mean_cat='cat',
                            all_hidden_states=self.config.output_hidden_states)#self.args.pool_type)
                            
        sizes = [self.pooler.get_sentence_embedding_dimension()] + list(map(int, self.args.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(Projector(config,sizes[i], sizes[i + 1]))
        layers.append(Projector(config,sizes[-2], sizes[-1]))
        self.projector = nn.Sequential(*layers)
        
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)
        
        self.apply(self._init_weights)

        self.bert = BertModel(self.config, add_pooling_layer=False).from_pretrained('bert-base-uncased')

        self._process_bert_encoder()
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        output = {}
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        bert_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=self.config.output_hidden_states,
            return_dict=return_dict,
        )
        if self.args.num_mixer_layers>0:
            mixer_output = self.nlpmixer(bert_output=bert_output,
                                    attention_mask=attention_mask)
            output['token_embeddings'] = mixer_output['token_embeddings']
            pooled = self.pooler(nlpmixer_output=mixer_output,attention_mask=attention_mask)
        else:
            pooled = self.pooler(bert_output=bert_output,attention_mask=attention_mask)
            output['token_embeddings'] = pooled['token_embeddings']

        projected = self.projector(pooled['sentence_embedding'])
        output['sentence_embedding'] = projected # Removed batchnorm for var_loss calc self.bn(projected)

        if self.args.mlm_weight != 0:
            assert mlm_input_ids is not None, "mlm_input_ids are needed for mlm"
            mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
            mlm_outputs = self.bert(
                mlm_input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=False, #True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
                return_dict=True,
            )
            prediction_scores = self.lm_head(mlm_outputs.last_hidden_state)
            output['prediction_scores'] = prediction_scores
        # pdb.set_trace()

        return output

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp
        
class BarlowBert(nn.Module):

    def __init__(self, config,args):
        super().__init__()

        self.config = config
        self.args = args
        self.model = SentenceBertWithNLPMixer(self.config,self.args)
        self.loss_fct = torch.nn.CrossEntropyLoss()

        self.sim = Similarity(self.args.temp)

    def forward(self,x):

        loss_dict = {}

        if self.args.mlm_weight != 0: # The input_ids are different due to mlm component
            y1, y2 = x
        else:
            y1=x
            y2=x
        # pdb.set_trace()
        output1 = self.model(**y1)
        output2 = self.model(**y2)

        projection1 = output1['sentence_embedding']
        projection2 = output2['sentence_embedding']

        dim = projection1.shape[1]
        diag = torch.eye(dim, device=projection1.device)

        if self.args.skip_barlow:
            loss = 0
        else:
            z1m = projection1 - projection1.mean(dim=0)
            z2m = projection2 - projection2.mean(dim=0)
            c_out = (z1m.T @ z2m) / (self.args.batch_size - 1)
            corr_loss = get_diag_loss(loss_dict,c_out,self.args.lambda_corr,'corr')
            loss_dict['corr_loss'] = corr_loss
            loss = corr_loss

        if self.args.cov_weight != 0:
            z1m = projection1 - projection1.mean(dim=0)
            z2m = projection2 - projection2.mean(dim=0)
            cov_z1 = (z1m.T @ z1m) / (self.args.batch_size - 1)
            cov_z2 = (z2m.T @ z2m) / (self.args.batch_size - 1)
            cov_loss = cov_z1[~diag.bool()].pow_(2).sum() / dim + cov_z2[~diag.bool()].pow_(2).sum() / dim #loss from off_diag entries
            loss = loss + self.args.cov_weight * cov_loss
            loss_dict['cov_loss'] = cov_loss
        
        if self.args.var_weight != 0:
            eps = 1e-4
            std_z1 = torch.sqrt(projection1.var(dim=0) + eps)
            std_z2 = torch.sqrt(projection2.var(dim=0) + eps)
            var_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))
            loss = loss + self.args.var_weight * var_loss
            loss_dict['var_loss'] = var_loss

        if self.args.simcse_weight != 0:
            cos_sim = self.sim(projection1.unsqueeze(1), projection2.unsqueeze(0))
            labels = torch.arange(cos_sim.size(0)).long().to(cos_sim.device)
            simcse_loss = self.loss_fct(cos_sim, labels)
            loss = loss + self.args.simcse_weight * simcse_loss
            loss_dict['simcse_loss'] = simcse_loss

        if self.args.mse_weight != 0:
            mse_loss = F.mse_loss(projection1, projection2)
            loss = loss + self.args.mse_weight * mse_loss
            loss_dict['mse_loss'] = mse_loss

        if self.args.sim_weight != 0:
            c_in = CosineSimilarity(projection1,projection2)
            # c_in.div_(self.args.batch_size)
            sim_loss = get_diag_loss(loss_dict,c_in,self.args.lambda_sim,'sim')
            loss = loss + self.args.sim_weight * sim_loss
            loss_dict['sim_loss'] = sim_loss

        if self.args.mlm_weight != 0:
            assert output1.get('prediction_scores') is not None
            prediction_scores = torch.cat([output1['prediction_scores'],output2['prediction_scores']],axis=1)
            mlm_labels = torch.cat([y1['mlm_labels'],y2['mlm_labels']],axis=1)
            mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
            masked_lm_loss = self.loss_fct(prediction_scores.view(-1, self.config.vocab_size), mlm_labels.view(-1))
            loss = loss + self.args.mlm_weight * masked_lm_loss
            loss_dict['mlm_loss'] = masked_lm_loss

        loss_dict['loss'] = loss
        return loss_dict        

class LARS2(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=False, lars_adaptation_filter=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)


    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self,closure=None):

        loss=None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    print(p.shape)
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])

        return loss

class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if g['weight_decay_filter'] is None or not g['weight_decay_filter'](p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if g['lars_adaptation_filter'] is None or not g['lars_adaptation_filter'](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])

def exclude_bias_and_norm(p):
    return p.ndim == 1
    
def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr * args.learning_rate_weights
    optimizer.param_groups[1]['lr'] = lr * args.learning_rate_biases


def main():
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    collator2 = DataCollatorForBarlowBertWithMLM(tokenizer=tokenizer)

    data = load_from_disk('/mounts/data/proj/jabbar/barlowbert/bookcorpus_20mil_128')
    loader = DataLoader(data,collate_fn=collator2,batch_size=32)
    batch = next(iter(loader))

    config = CONFIG_MAPPING['bert'].from_pretrained('bert-base-uncased')
    config.max_position_embeddings=128
    config.output_hidden_states = True
    Args = namedtuple('Args',['do_mlm', 'do_sim','batch_size', 'lambda_corr'])
    args = Args(False, False, 4, 0.1)
    model_nlp = SentenceBertWithNLPMixer(config,args)
    model_bb = BarlowBert(config,args)

    out = model_bb(batch)
    print(out)

if __name__ == "__main__":
    main()