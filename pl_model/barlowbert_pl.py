from collections import namedtuple
import pdb
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import CONFIG_MAPPING, BertTokenizerFast, AdamW, get_linear_schedule_with_warmup
import torch
from argparse import ArgumentParser
from pl_bolts.optimizers import LARS

import pytorch_lightning as pl
from barlowbert_models import BarlowBert
from barlowbert_dm import BookCorpusDataModuleForMLM, DataCollatorForBarlowBertWithMLM

bert_small = {
    "hidden_size" : 512 ,
    "num_hidden_layers" : 4,
    "num_attention_heads": int(512/64),
    "intermediate_size" : int(512*4)
}

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

        if self.args.do_mlm:
            self.log('masked_lm_loss',self.args.mlm_weight * loss_dict['mlm_loss'])

        if self.args.do_sim:
            self.log("sim_loss",loss_dict['sim_loss'])
            self.log("sim_ondiag",loss_dict['sim_ondiag'])
            self.log("sim_offdiag",loss_dict['sim_offdiag'])   

        # pdb.set_trace()
        self.log("train_loss", loss_dict['loss'])
        self.log("corr_loss",loss_dict['corr_loss'])
        self.log("corr_ondiag",loss_dict['corr_ondiag'])
        self.log("corr_offdiag",loss_dict['corr_offdiag'])
        self.log("lr",self.optimizers().param_groups[0]['lr'],prog_bar=True)        

        # tensorboard = self.logger.experiment[0]
        # if self.global_step % 500 == 0:
        #     tensorboard.add_image('correlation_matrix',c_out,global_step=self.global_step,dataformats='HW')
        #     tensorboard.add_image('similarity_matrix',c_in,global_step=self.global_step,dataformats='HW')
        
        return loss_dict['loss']

    def configure_optimizers(self):
        # if self.args.dont_use_lars:
        #     return torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)            
        # else:
        #     return LARS(self.parameters(), lr=self.args.lr,weight_decay=self.args.weight_decay)
        if self.args.dont_use_lars:
            optim = torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)    
        else:
            optim = LARS(self.parameters(), lr=self.args.lr,weight_decay=self.args.weight_decay)
        
        sched = torch.optim.lr_scheduler.OneCycleLR(optim,max_lr=self.args.lr,total_steps=self.num_training_steps,anneal_strategy='linear')
        # sched = get_linear_schedule_with_warmup(
        #     optimizer = optim,
        #     num_warmup_steps = int(self.args.warmup_ratio * self.num_training_steps),
        #     num_training_steps = self.num_training_steps
        # )

        lr_scheduler_config = {
            # REQUIRED: The scheduler instance
            "scheduler": sched,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "step",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            # "monitor": None,
            # If set to `True`, will enforce that the value specified 'monitor'
            # is available when the scheduler is updated, thus stopping
            # training if not found. If set to `False`, it will only produce a warning
            # "strict": True,
            # If using the `LearningRateMonitor` callback to monitor the
            # learning rate progress, this keyword can be used to specify
            # a custom logged name
            "name": 'lr_scheduler',
        }
        return {'optimizer':optim,
                'scheduler':lr_scheduler_config
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
        parser.add_argument('--all_hidden_states', action='store_true')
        parser.add_argument('--do_mlm', action='store_true')
        parser.add_argument('--do_sim', action='store_true')
        parser.add_argument('--dont_use_bert', action='store_true')
        parser.add_argument('--dont_use_lars', action='store_true')
        parser.add_argument('--mlm_weight', type=float, default=0.1)
        parser.add_argument('--warmup_ratio', type=float, default=0.1)
        parser.add_argument('--num_mixer_layers', type=int, default=0)
        parser.add_argument('--num_trainable_layers', type=int, default=6)
        parser.add_argument('--bert_pooler', type=bool, default=False)
        parser.add_argument('--max_pooling', type=bool, default=False)
        parser.add_argument('--mean_pooling', type=bool, default=True)
        parser.add_argument('--cls_pooling', type=bool, default=False)
        parser.add_argument('--projector', default='768-256-256', type=str,
                        metavar='MLP', help='projector MLP')
        parser.add_argument('--pool_type', type=str,
                        default='cat')
        parser.add_argument('--hidden_dropout_prob', 
                        default=0.1, type=float,
                        help='dropout for hidden layers')
        parser.add_argument('--lambda_corr', type=float, default=0.001)
        parser.add_argument('--lambda_sim', type=float, default=1.0)
        parser.add_argument('--sim_weight', type=float, default=0.001)               
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

    tmp_args = '--fast_dev_run True --exp_name bert --gpus 1 --dataset 20mil --precision 16 --batch_size 32 --all_hidden_states --num_trainable_layers 0'.split()
    tmp_args_2 = "--gpus 1 --projector 2048-2048 --dataset 20mil --precision 16 --log_every_n_steps 20 --do_sim --tags frozen 2048-2048 gpus1 sim".split()    
    args = parser.parse_args()

    args.tags.insert(0, args.exp_name)
    args.accelerator = 'ddp'
    args.benchmark = True

    return args

def main():

    args = args_parse()
    pl.seed_everything(args.seed)

    if args.exp_name=='bert_small':
        config = CONFIG_MAPPING['bert'].from_pretrained('prajjwal1/bert-small')
    else:
        config = CONFIG_MAPPING['bert'].from_pretrained('bert-base-uncased')
    
    if args.all_hidden_states:
        config.output_hidden_states=True

    config.max_position_embeddings=128 #because of the preprocessed dataset
    config.hidden_dropout_prob = args.hidden_dropout_prob

    model = LitBarlowBert(args,config)
    dm = BookCorpusDataModuleForMLM(args)

    tb_logger = TensorBoardLogger(
                                    save_dir=args.datadir,
                                    version='_'.join(args.tags),
                                    name='lightning_tb_logs',
                                    )

    ckpt_callback = ModelCheckpoint(dirpath=args.datadir/'checkpoint/'/'_'.join(args.tags),
                                    # filename='_'.join(args.tags),
                                    every_n_train_steps=4000,
                                    save_top_k=-1,
                                    # train_time_interval=timedelta(hours=4)
                                    )

    trainer = pl.Trainer.from_argparse_args(args,
                                            plugins=pl.plugins.DDPPlugin(find_unused_parameters=False),
                                            logger=[tb_logger],
                                            callbacks=[ckpt_callback]
                                            )

    trainer.fit(model, dm)

if __name__=='__main__':
    main()