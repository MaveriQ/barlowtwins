from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import CONFIG_MAPPING, BertTokenizer
import torch
from argparse import ArgumentParser
import pytorch_lightning as pl
from barlowbert_models import BarlowBert, off_diagonal, LARS, adjust_learning_rate
from barlowbert_dm import BookCorpusDataModuleForMLM

bert_small = {
    "hidden_size" : 512 ,
    "num_hidden_layers" : 4,
    "num_attention_heads": int(512/64),
    "intermediate_size" : int(512*4)
}

bert_tiny = {
    "hidden_size" : 128 ,
    "num_hidden_layers" : 2,
    "num_attention_heads": int(128/64),
    "intermediate_size" : int(128*4)
}

class LitBarlowBert(pl.LightningModule):
    def __init__(self, args,config):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.model = BarlowBert(config)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        y1, y2 = batch

        output1 = self.model(**y1)
        output2 = self.model(**y2)

        c = (output1.transpose(0,1) @ output2)
        # c.div_(self.args.batch_size)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag   

        self.log("train_loss", loss)
        self.log("on_diag", on_diag)
        self.log("off_diag", off_diag)
        
        return loss

    def configure_optimizers(self):
        param_weights = []
        param_biases = []
        for param in self.parameters():
            if param.ndim == 1:
                param_biases.append(param)
            else:
                param_weights.append(param)

        parameters = [{'params': param_weights}, {'params': param_biases}]
        # optimizer = LARS(parameters, lr=0, weight_decay=self.args.weight_decay,
        #              weight_decay_filter=True,
        #              lars_adaptation_filter=True)

        return torch.optim.AdamW(self.parameters(), lr=self.args.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False) 
        parser.add_argument('--lambd', type=float, default=0.005)
        parser.add_argument('--max_pooling', type=bool, default=True)
        parser.add_argument('--mean_pooling', type=bool, default=True)
        parser.add_argument('--cls_pooling', type=bool, default=True)
        parser.add_argument('--projector', default='128-128', type=str,
                        metavar='MLP', help='projector MLP')               
        return parser

def args_parse():
    parser = ArgumentParser(description='Barlow Twins Training')
    parser.add_argument('--exp_name', 
                        type=str,
                        default='bert_small')
    parser.add_argument('--tags', 
                        nargs='*',
                        type=str,
                        default=[])
    parser.add_argument('--seed',
                        type=int,
                        default=42)
    parser.add_argument('--workers', 
                        default=32, type=int, metavar='N',
                        help='number of data loader workers')
    parser.add_argument('--epochs', 
                        default=5, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr', 
                        default=0.02, type=float, metavar='LR',
                        help='base learning rate for weights')
    # parser.add_argument('--learning-rate-weights', default=0.2, type=float, metavar='LR',
    #                     help='base learning rate for weights')
    # parser.add_argument('--learning-rate-biases', default=0.0048, type=float, metavar='LR',
    #                     help='base learning rate for biases and batch norm parameters')
    parser.add_argument('--weight-decay', 
                        default=1e-6, type=float, metavar='W',
                        help='weight decay')
    parser.add_argument('--print-freq', 
                        default=100, type=int, metavar='N',
                        help='print frequency')
    parser.add_argument('--checkpoint-dir', 
                        default='./checkpoint/', type=Path,
                        metavar='DIR', help='path to checkpoint directory')

    parser = LitBarlowBert.add_model_specific_args(parser)
    parser = BookCorpusDataModuleForMLM.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    tmp_args = '--fast_dev_run False --gpus 2 --batch_size 128 --lr 0.01 --accelerator ddp --benchmark True'.split()    
    args = parser.parse_args()

    args.tags.insert(0, args.exp_name)

    return args

if __name__=='__main__':

    args = args_parse()
    pl.seed_everything(args.seed)
    config = CONFIG_MAPPING['bert']()
    config.update(bert_small)
    config.projector=args.projector
    config.max_pooling=args.max_pooling
    config.mean_pooling=args.mean_pooling
    config.cls_pooling=args.cls_pooling

    model = LitBarlowBert(args,config)
    dm = BookCorpusDataModuleForMLM(args)

    tb_logger = TensorBoardLogger(
                                    save_dir=args.datadir,
                                    version='_'.join(args.tags),
                                    name='lightning_tb_logs'
                                    )

    trainer = pl.Trainer.from_argparse_args(args,
                                            plugins=pl.plugins.DDPPlugin(find_unused_parameters=False),
                                            logger=[tb_logger],
                                            )

    trainer.fit(model, dm)