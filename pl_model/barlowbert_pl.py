from collections import namedtuple
import pdb
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import CONFIG_MAPPING, BertTokenizerFast, AdamW, get_linear_schedule_with_warmup
import torch
from argparse import ArgumentParser
from pl_bolts.optimizers import LARS
from pytorch_lightning.callbacks import LearningRateMonitor
import pytorch_lightning as pl
from barlowbert_models import BarlowBert
from barlowbert_dm import BookCorpusDataModuleForMLM, DataCollatorForBarlowBertWithMLM

bert_small = {
    "hidden_size" : 512 ,
    "num_hidden_layers" : 4,
    "num_attention_heads": int(512/64),
    "intermediate_size" : int(512*4)
}

def bert_base_AdamW_LLRD(cls,mul_factor=0.5):

    num_layers = cls.args.num_trainable_layers
    
    opt_parameters = []    # To be passed to the optimizer (only parameters of the layers you want to update).
    named_parameters = list(cls.model.named_parameters()) 
        
    # According to AAAMLP book by A. Thakur, we generally do not use any decay 
    # for bias and LayerNorm.weight layers.
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    init_lr = cls.args.lr 
    head_lr = init_lr
    lr = init_lr
    
    # === Projector ======================================================  
    
    params_0 = [p for n,p in named_parameters if ("projector" in n) 
                and any(nd in n for nd in no_decay)]
    params_1 = [p for n,p in named_parameters if ("projector" in n)
                and not any(nd in n for nd in no_decay)]
    
    head_params = {"params": params_0, "lr": head_lr, "weight_decay": 0.0}    
    opt_parameters.append(head_params)
        
    head_params = {"params": params_1, "lr": head_lr, "weight_decay": 0.01}    
    opt_parameters.append(head_params)
                
    # === num_layers Hidden layers ==========================================================
    
    for layer in range(11,11-num_layers,-1):        
        params_0 = [p for n,p in named_parameters if f"encoder.layer.{layer}." in n 
                    and any(nd in n for nd in no_decay)]
        params_1 = [p for n,p in named_parameters if f"encoder.layer.{layer}." in n 
                    and not any(nd in n for nd in no_decay)]
        
        layer_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
        opt_parameters.append(layer_params)   
                            
        layer_params = {"params": params_1, "lr": lr, "weight_decay": 0.01}
        opt_parameters.append(layer_params)       
        
        lr *= mul_factor     
        
    # === Embeddings layer ==========================================================
    
    params_0 = [p for n,p in named_parameters if "embeddings" in n 
                and any(nd in n for nd in no_decay)]
    params_1 = [p for n,p in named_parameters if "embeddings" in n
                and not any(nd in n for nd in no_decay)]
    
    embed_params = {"params": params_0, "lr": lr, "weight_decay": 0.0} 
    opt_parameters.append(embed_params)
        
    embed_params = {"params": params_1, "lr": lr, "weight_decay": 0.01} 
    opt_parameters.append(embed_params)        
    
    return AdamW(opt_parameters, lr=init_lr)

class LitBarlowBert(pl.LightningModule):
    def __init__(self, args,config):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.config = config
        self.model = BarlowBert(self.config,self.args)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        output = self.model.model(**x)
        return output['sentence_embedding']
        
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward

        loss_dict = self.model(batch)

        for key,val in loss_dict.items():
            self.log(key,val)

        # tensorboard = self.logger.experiment[0]
        # if self.global_step % 500 == 0:
        #     tensorboard.add_image('correlation_matrix',c_out,global_step=self.global_step,dataformats='HW')
        #     tensorboard.add_image('similarity_matrix',c_in,global_step=self.global_step,dataformats='HW')
        
        return loss_dict['loss']

    def configure_optimizers(self):
        # param_weights = []
        # param_biases = []
        # for param in self.parameters():
        #     if param.ndim == 1:
        #         param_biases.append(param)
        #     else:
        #         param_weights.append(param)

        # parameters = [{'params': param_weights}, {'params': param_biases}]
        # # optimizer = LARS2(parameters, lr=0, weight_decay=self.args.weight_decay,
        # #              weight_decay_filter=True,
        # #              lars_adaptation_filter=True)
        # optimizer = LARS(model.parameters(), lr=0, weight_decay=args.weight_decay,
        #              weight_decay_filter=exclude_bias_and_norm,
        #              lars_adaptation_filter=exclude_bias_and_norm)

        if self.args.skip_lars:
            optim = torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        # elif self.args.use_llrd:
        #     optim = bert_base_AdamW_LLRD(self)            
        else:
            optim = LARS(self.parameters(), lr=self.args.lr,weight_decay=self.args.weight_decay)
        
        # sched = torch.optim.lr_scheduler.OneCycleLR(optim,max_lr=self.args.lr,total_steps=self.num_training_steps,anneal_strategy='linear')
        # sched = get_linear_schedule_with_warmup(
        #     optimizer = optim,
        #     num_warmup_steps = int(self.args.warmup_ratio * self.num_training_steps),
        #     num_training_steps = self.num_training_steps
        # )

        # lr_scheduler_config = {
        #     # REQUIRED: The scheduler instance
        #     "scheduler": sched,
        #     # The unit of the scheduler's step size, could also be 'step'.
        #     # 'epoch' updates the scheduler on epoch end whereas 'step'
        #     # updates it after a optimizer update.
        #     "interval": "step",
        #     # How many epochs/steps should pass between calls to
        #     # `scheduler.step()`. 1 corresponds to updating the learning
        #     # rate after every epoch/step.
        #     "frequency": 1,
        #     # Metric to to monitor for schedulers like `ReduceLROnPlateau`
        #     # "monitor": None,
        #     # If set to `True`, will enforce that the value specified 'monitor'
        #     # is available when the scheduler is updated, thus stopping
        #     # training if not found. If set to `False`, it will only produce a warning
        #     # "strict": True,
        #     # If using the `LearningRateMonitor` callback to monitor the
        #     # learning rate progress, this keyword can be used to specify
        #     # a custom logged name
        #     "name": 'lr_scheduler',
        # }
        return {'optimizer':optim,
                # 'scheduler':lr_scheduler_config
                }        

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())
        batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(limit_batches * batches)     

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs
        
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--skip_hidden_states', action='store_true')

        parser.add_argument('--skip_barlow', action='store_true')
        parser.add_argument('--simcse_weight', type=float, default=0)
        parser.add_argument('--cov_weight', type=float, default=0)
        parser.add_argument('--var_weight', type=float, default=0)
        parser.add_argument('--mse_weight', type=float, default=0)
        parser.add_argument('--sim_weight', type=float, default=0)
        parser.add_argument('--mlm_weight', type=float, default=0)
        parser.add_argument('--lambda_corr', type=float, default=0.01)
        parser.add_argument('--lambda_sim', type=float, default=1.0)   

        parser.add_argument('--temp', type=float, default=0.05) 
        # parser.add_argument('--skip_bert', action='store_true')
        parser.add_argument('--skip_lars', action='store_true')
        parser.add_argument('--warmup_ratio', type=float, default=0.1)
        parser.add_argument('--num_mixer_layers', type=int, default=0)
        parser.add_argument('--num_trainable_layers', type=int, default=3)

        parser.add_argument('--bert_pooler', type=bool, default=False)
        parser.add_argument('--max_pooling', type=bool, default=False)
        parser.add_argument('--mean_pooling', type=bool, default=True)
        parser.add_argument('--cls_pooling', type=bool, default=False)

        parser.add_argument('--projector', default='768-256-256', type=str)
        parser.add_argument('--pool_type', type=str, default='mean')

        parser.add_argument('--hidden_dropout_prob', default=0.1, type=float, help='dropout for hidden layers')
        parser.add_argument('--attention_probs_dropout_prob', default=0.1, type=float, help='dropout for attention layers')           
        return parser

def args_parse():
    parser = ArgumentParser(description='Barlow Twins Training')
    parser.add_argument('--exp_name', 
                        type=str,
                        default='bert')
    parser.add_argument('--tags', 
                        nargs='*',
                        type=str,
                        default=[])
    parser.add_argument('--seed',
                        type=int,
                        default=42)
    parser.add_argument('--lr', 
                        default=0.001, type=float, metavar='LR',
                        help='base learning rate for weights')
    parser.add_argument('--weight-decay', 
                        default=1e-6, type=float, metavar='W',
                        help='weight decay')

    parser = LitBarlowBert.add_model_specific_args(parser)
    parser = BookCorpusDataModuleForMLM.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    tmp_args = '--fast_dev_run True --exp_name bert --gpus 1 --var_weight 1.0 --cov_weight 1.0 --simcse_weight 1.0 --mse_weight 1.0'.split()
    tmp_args_2 = "--gpus 1 --precision 16 --batch_size 16 --num_trainable_layers 3 --var_weight 1.0 --cov_weight 1.0 --simcse_weight 1.0 --mse_weight 1.0 --seq_len 512 --projector 768 --max_epochs 1".split()    
    args = parser.parse_args()

    args.tags.insert(0, args.exp_name)
    args.precision = 16
    args.accelerator = 'ddp'
    # args.plugins = []
    args.benchmark = True

    return args

def main():

    args = args_parse()
    pl.seed_everything(args.seed)

    if args.exp_name=='bert_small':
        config = CONFIG_MAPPING['bert'].from_pretrained('prajjwal1/bert-small')
    else:
        config = CONFIG_MAPPING['bert'].from_pretrained('bert-base-uncased')
    
    if args.skip_hidden_states:
        config.output_hidden_states=False
    else:
        config.output_hidden_states=True

    config.max_position_embeddings=128 #because of the preprocessed dataset
    config.hidden_dropout_prob = args.hidden_dropout_prob

    model = LitBarlowBert(args,config)
    # pdb.set_trace()
    dm = BookCorpusDataModuleForMLM(args)

    tb_logger = TensorBoardLogger(
                                    save_dir=args.datadir,
                                    version='_'.join(args.tags),
                                    name='lightning_tb_logs',
                                    )

    ckpt_callback = ModelCheckpoint(dirpath=args.datadir/'checkpoint'/'_'.join(args.tags),
                                    # filename='_'.join(args.tags),
                                    every_n_train_steps=2000,
                                    save_top_k=-1,
                                    # train_time_interval=timedelta(hours=4)
                                    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer.from_argparse_args(args,
                                            plugins=pl.plugins.DDPPlugin(find_unused_parameters=False),
                                            logger=[tb_logger],
                                            callbacks=[ckpt_callback]# lr_monitor]
                                            )

    trainer.fit(model, dm)

if __name__=='__main__':
    main()