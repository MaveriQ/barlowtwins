import logging
import sys
sys.path.insert(0, '..')
from barlowbert_pl import LitBarlowBert
from transformers import BertTokenizerFast
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
import pickle
import os
import torch
import pdb
import json
from prettytable import PrettyTable
import pandas as pd

list_of_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                    'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                    'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
                    'Length', 'WordContent', 'Depth', 'TopConstituents',
                    'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                    'OddManOut', 'CoordinationInversion']

PATH_TO_DATA = '/mounts/Users/cisintern/jabbar/git/SentEval/data'
PATH_TO_SENTEVAL = '/mounts/Users/cisintern/jabbar/git/SimCSE/SentEval'
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

def args_parse():
    parser = ArgumentParser(description='Barlow Twins Evaluation')

    parser.add_argument('--checkpoint_dir', 
                        type=Path,
                        required=True,
                        default='')    
    parser.add_argument('--checkpoint', 
                        type=str,
                        required=False,
                        default='')
    parser.add_argument('--result_filename', 
                        type=str,
                        default='_results')
    parser.add_argument("--task_set", type=str, 
            choices=['sts', 'sts_dev', 'transfer', 'full', 'na'],
            default='sts',
            help="What set of tasks to evaluate on. If not 'na', this will override '--tasks'")
    parser.add_argument("--mode", type=str, 
            choices=['dev', 'test', 'fasttest'],
            default='dev', 
            help="What evaluation mode to use (dev: fast mode, dev results; test: full mode, test results); fasttest: fast mode, test results")
    parser.add_argument('--tasks', 
                        nargs='+',
                        type=str,
                        default=['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                                 'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC',
                                 'SICKRelatedness', 'STSBenchmark'],
                        choices=list_of_tasks)

    tmp_args = "--checkpoint_dir /mounts/data/proj/jabbar/barlowbert/checkpoint/bert_frozen_bs1024_dropout0.1_lr1e-3_lambd1e-3_mlm0_gpus4_mean_all_fp16".split()
    args = parser.parse_args()
    return args

def prepare(params, samples):
    return

def batcher(params, batch):
    # many snippets are from : https://github.com/princeton-nlp/SimCSE/blob/main/evaluation.py
    
    if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
        print('caught it')
        batch = [[word.decode('utf-8') for word in s] for s in batch]

    sentences = [' '.join(s) for s in batch]
    
    tokens = params.tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding='max_length',
                max_length=128,
                return_token_type_ids=False,
                truncation=True
            )
    
    for k in tokens:
            tokens[k] = tokens[k].to(params.device)
    
    with torch.no_grad():
        embedding = params.model(tokens)

    return embedding.detach().cpu().numpy()

def list_to_dict(listofdicts):
    nd={}
    for d in listofdicts:
        for k,v in d.items():
            try:
                nd[k].append(v)
            except KeyError:
                nd[k]=[v]
    return nd

def add_pool_type(args):
    data = torch.load(args.checkpoint)
    
    data['hyper_parameters']['config'].pool_type='cat'
    torch.save(data,args.checkpoint)

def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)
    
def main(args):
    
    # add_pool_type(args)

    # Set up the tasks
    if args.task_set == 'sts':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
    if args.task_set == 'sts_dev':
        args.tasks = ['STSBenchmark', 'SICKRelatedness']
    elif args.task_set == 'transfer':
        args.tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']
    elif args.task_set == 'full':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        args.tasks += ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']

    # Set params for SentEval
    if args.mode == 'dev' or args.mode == 'fasttest':
        # Fast mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                         'tenacity': 3, 'epoch_size': 2}
    elif args.mode == 'test':
        # Full mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
        params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                         'tenacity': 5, 'epoch_size': 4}
    else:
        raise NotImplementedError
        
    params['tokenizer'] = BertTokenizerFast.from_pretrained('bert-base-uncased', padding=True,truncation=True,)
    params['model'] = LitBarlowBert.load_from_checkpoint(args.checkpoint)
    params['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    params['model'] = params['model'].to(params['device'])

    # Set up logger
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.ERROR)

#     se = senteval.engine.SE(params, batcher, prepare)
#     results = se.eval(args.transfer_tasks)
    results = {}

    for task in args.tasks:
        se = senteval.engine.SE(params, batcher, prepare)
        result = se.eval(task)
        results[task] = result
    
   # Print evaluation results
    if args.mode == 'dev':
#         pdb.set_trace()
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in ['STSBenchmark', 'SICKRelatedness']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['dev']['spearman'][0] * 100))
            else:
                scores.append("0.00")
        # print_table(task_names, scores)

        # task_names = []
        # scores = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['devacc']))    
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)

    elif args.mode == 'test' or args.mode == 'fasttest':
#         pdb.set_trace()
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
            task_names.append(task)
            if task in results:
                if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                    scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100)) 
                else:
                    scores.append("%.2f" % (results[task]['test']['spearman'].correlation * 100)) 
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        # print_table(task_names, scores)

        # task_names = []
        # scores = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['acc']))    
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)
    
    return (task_names, scores)

        
    # args.result_filename = args.checkpoint + args.result_filename + '.pkl'
    # i=0
    # while os.path.exists(args.result_filename):
    #     i+=1
    #     args.result_filename = args.result_filename + f'_{i}.pkl'

    # with open(args.result_filename,'w') as file:
    #     json.dump(results,file,indent=4)

if __name__=='__main__':
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
    args = args_parse()

    all_checkpoints = list(Path(args.checkpoint_dir).glob('*.ckpt'))

    filename = args.checkpoint_dir.name+'.csv'
    if os.path.exists(filename):
        all_results = pd.read_csv(filename)
    else:
        all_results = pd.DataFrame()

    if len(all_checkpoints)==0:
        print(f'No checkpoints found at {args.checkpoint_dir}')
        sys.exit(1)
    else:
        print(f'Processing {len(all_checkpoints)} checkpoints...')
    # pdb.set_trace()
    for ckpt in all_checkpoints:
        epoch,step=ckpt.stem.split('-')
        epoch = epoch.split('=')[1]
        step = step.split('=')[1]
        if len(all_results)>0: # loaded some results
            if (int(epoch) in all_results['epoch'].values) & (int(step) in all_results['step'].values):
                print(f'Skipping epoch {epoch}, step {step}.')
                continue
        args.checkpoint = str(ckpt)
        task_names, scores = main(args)

        result_row = dict(zip(task_names,scores))
        result_row['epoch']=epoch
        result_row['step']=step
        all_results = all_results.append(result_row,ignore_index=True)

    all_results.to_csv(filename)
        
