# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os
import logging
import random
import warnings
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

from semilearn.algorithms import get_algorithm, name2alg
from semilearn.imb_algorithms import get_imb_algorithm, name2imbalg
from semilearn.core.utils import get_net_builder, get_logger, get_port, send_model_cuda, count_parameters, over_write_args_from_file, TBLog


def get_config():
    from semilearn.algorithms.utils import str2bool

    parser = argparse.ArgumentParser(description='Semi-Supervised Learning (USB)')

    '''
    Saving & loading of the model.
    '''
    parser.add_argument('--gpu_id', default='None', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--run_id', type=str, help='id(s) for runs')
    parser.add_argument('--seed', default=None, type=int, help="random seed")
    # parser.add_argument('--save_dir', default='/scr/chrisyi/PivotMatch_MSFT', help='directory to output the result')
    parser.add_argument('--data_dir', default='/scr/chrisyi/data', help='directory to scr data')
    parser.add_argument('--save_dir', default='output', help='directory to output the result')
    # parser.add_argument('--save_dir', default='/data/chrisyi/PivotMatch', help='directory to output the result')
    # parser.add_argument('--data_dir', default='/data/chrisyi/data', help='directory to scr data')
    parser.add_argument('--overwrite', action='store_true', help="overwrite folder")

    # parser.add_argument('--save_dir', type=str, default='./saved_models')
    # parser.add_argument('-sn', '--save_name', type=str, default='fixmatch')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--load_path', type=str)
    # parser.add_argument('-o', '--overwrite', action='store_true', default=True)
    # parser.add_argument('--use_tensorboard', action='store_true', help='Use tensorboard to plot and save curves')
    # parser.add_argument('--use_wandb', action='store_true', help='Use wandb to plot and save curves')
    parser.add_argument('--use_aim', action='store_true', help='Use aim to plot and save curves')

    '''
    Training Configuration of FixMatch
    '''
    # parser.add_argument('--epoch', type=int, default=1)
    # parser.add_argument('--num_train_iter', type=int, default=20, help='total number of training iterations')
    parser.add_argument('--num_warmup_iter', type=int, default=0, help='cosine linear warmup iterations')
    # parser.add_argument('--num_eval_iter', type=int, default=10, help='evaluation frequency')
    # parser.add_argument('--num_log_iter', type=int, default=5, help='logging frequencu')
    # parser.add_argument('-nl', '--num_labels', type=int, default=400)
    # parser.add_argument('-bsz', '--batch_size', type=int, default=8)
    # parser.add_argument('--uratio', type=int, default=1, help='the ratio of unlabeled data to labeld data in each mini-batch')
    # parser.add_argument('--eval_batch_size', type=int, default=16, help='batch size of evaluation data loader (it does not affect the accuracy)')
    # parser.add_argument('--ema_m', type=float, default=0.999, help='ema momentum for eval_model')
    parser.add_argument('--ulb_loss_ratio', type=float, default=1.0)

    '''
    Optimizer configurations
    '''
    # parser.add_argument('--optim', type=str, default='SGD')
    # parser.add_argument('--lr', type=float, default=3e-2)
    # parser.add_argument('--momentum', type=float, default=0.9)
    # parser.add_argument('--weight_decay', type=float, default=5e-4)
    # parser.add_argument('--layer_decay', type=float, default=1.0, help='layer-wise learning rate decay, default to 1.0 which means no layer decay')

    '''
    Backbone Net Configurations
    '''
    # parser.add_argument('--net', type=str, default='wrn_28_2')
    # parser.add_argument('--net_from_name', type=str2bool, default=False)
    parser.add_argument('--use_pretrain', default=False, type=str2bool)
    parser.add_argument('--pretrain_path', default='', type=str)

    '''
    Algorithms Configurations
    '''  

    ## core algorithm setting
    # parser.add_argument('-alg', '--algorithm', type=str, default='fixmatch', help='ssl algorithm')
    # parser.add_argument('--use_cat', type=str2bool, default=True, help='use cat operation in algorithms')
    # parser.add_argument('--amp', type=str2bool, default=False, help='use mixed precision training or not')
    parser.add_argument('--clip_grad', type=float, default=0)

    ## imbalance algorithm setting
    parser.add_argument('-imb_alg', '--imb_algorithm', type=str, default=None, help='imbalance ssl algorithm')

    '''
    Data Configurations
    '''

    ## standard setting configurations
    # # parser.add_argument('--data_dir', type=str, default='./data')
    # parser.add_argument('-ds', '--dataset', type=str, default='cifar10')
    # parser.add_argument('-nc', '--num_classes', type=int, default=10)
    # parser.add_argument('--train_sampler', type=str, default='RandomSampler')
    # parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--include_lb_to_ulb', type=str2bool, default='True', help='flag of including labeled data into unlabeled data, default to True')

    ## imbalanced setting arguments
    parser.add_argument('--lb_imb_ratio', type=int, default=1, help="imbalance ratio of labeled data, default to 1")
    parser.add_argument('--ulb_imb_ratio', type=int, default=1, help="imbalance ratio of unlabeled data, default to 1")
    parser.add_argument('--ulb_num_labels', type=int, default=None, help="number of labels for unlabeled data, used for determining the maximum number of labels in imbalanced setting")

    # ## cv dataset arguments
    # parser.add_argument('--img_size', type=int, default=32)
    # parser.add_argument('--crop_ratio', type=float, default=0.875)

    # ## nlp dataset arguments 
    # parser.add_argument('--max_length', type=int, default=512)

    # ## speech dataset algorithms
    # parser.add_argument('--max_length_seconds', type=float, default=4.0)
    # parser.add_argument('--sample_rate', type=int, default=16000)

    '''
    multi-GPUs & Distrbitued Training
    '''

    ## args for distributed training (from https://github.com/pytorch/examples/blob/master/imagenet/main.py)
    parser.add_argument('--world-size', default=1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int, help='**node rank** for distributed training')
    parser.add_argument('-du', '--dist-url', default='tcp://127.0.0.1:11111', type=str, help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    # parser.add_argument('--seed', default=1, type=int, help='seed for initializing training. ')
    # parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', type=str2bool, default=False,
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    # config file
    parser.add_argument('--cfg', type=str, default='')

    # add algorithm specific parameters
    args = parser.parse_args()
    over_write_args_from_file(args, args.cfg)
    print(name2alg.keys)
    for argument in name2alg[args.algorithm].get_argument():
        parser.add_argument(argument.name, type=argument.type, default=argument.default, help=argument.help)

    # add imbalanced algorithm specific parameters
    args = parser.parse_args()
    over_write_args_from_file(args, args.cfg)
    if args.imb_algorithm is not None:
        for argument in name2imbalg[args.imb_algorithm].get_argument():
            parser.add_argument(argument.name, type=argument.type, default=argument.default, help=argument.help)
    args = parser.parse_args()
    over_write_args_from_file(args, args.cfg)
    return args


def main(args):
    '''
    For (Distributed)DataParallelism,
    main(args) spawn each process (main_worker) to each GPU.
    '''
    if args.run_id[:9] == 'reproduce':
        args.save_dir = args.save_dir + '/' + args.algorithm + '/' + args.dataset + '/' + str(args.num_labels) + '/' + 'reproduce'  + '/' + 'seed'+ '_' + str(args.seed) 
    else:
        args.save_dir = args.save_dir + '/' + args.algorithm + '/' + args.dataset + '/' + str(args.num_labels) + '/' + 'exp'+ '_' + str(args.run_id)  + '/' + 'seed'+ '_' + str(args.seed) 

    args.save_name = 'checkpoints'

    assert args.num_train_iter % args.epoch == 0, f"# total training iter. {args.num_train_iter} is not divisible by # epochs {args.epoch}"

    # save_path = os.path.join(args.save_dir, args.save_name)
    save_path = args.save_dir
    if os.path.exists(save_path) and args.overwrite and args.resume == False:
        import shutil
        shutil.rmtree(save_path)
    if os.path.exists(save_path) and not args.overwrite:
        raise Exception('already existing model: {}'.format(save_path))
    if args.resume:
        if args.load_path is None:
            raise Exception('Resume of training requires --load_path in the args')
        if os.path.abspath(save_path) == os.path.abspath(args.load_path) and not args.overwrite:
            raise Exception('Saving & Loading paths are same. \
                            If you want over-write, give --overwrite in the argument.')

    if args.seed is not None:
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    args.gpu = args.gpu_id 
    if args.gpu == 'None':
        args.gpu = None
    if args.gpu is not None:
        args.gpu = int(args.gpu_id) 
        args.gpu_id = int(args.gpu_id) 
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    # distributed: true if manually selected or if world_size > 1
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.distributed = False
    ngpus_per_node = torch.cuda.device_count()  # number of gpus of each node

    if args.rank % ngpus_per_node == 0:
        # os.makedirs(args.out, exist_ok=True)
        os.makedirs(args.save_dir, exist_ok=args.overwrite)
        # cfg.dump(os.path.join(args.out, os.path.basename(args.cfg)))
        with open('{}/args.txt'.format(args.save_dir), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        # with open('{}/configs.txt'.format(args.out), 'w') as f:
        #     json.dump(cfg.__dict__, f, indent=2)

    if args.multiprocessing_distributed:
        # now, args.world_size means num of total processes in all nodes
        args.world_size = ngpus_per_node * args.world_size

        # args=(,) means the arguments of main_worker
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    '''
    main_worker is conducted on each GPU.
    '''

    global best_acc1
    # args.gpu_id = gpu

    # random seed has to be set for the synchronization of labeled data sampling in each process.
    assert args.seed is not None
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # cudnn.deterministic = True
    cudnn.benchmark = True

    # SET UP FOR DISTRIBUTED TRAINING
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])

        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu  # compute global rank

        # set distributed group:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # SET save_path and logger
    # save_path = os.path.join(args.save_dir, args.save_name)
    save_path = args.save_dir
    logger_level = "WARNING"
    tb_log = None
    if args.rank % ngpus_per_node == 0:
        tb_log = TBLog(save_path, 'tensorboard', use_tensorboard=args.use_tensorboard)
        logger_level = "INFO"

    # logger = get_logger(args.save_name, save_path, logger_level)
    logger = get_logger('logger', save_path, logger_level)
    logger.info(f"Use GPU: {args.gpu} for training")

    _net_builder = get_net_builder(args.net, args.net_from_name)
    # optimizer, scheduler, datasets, dataloaders with be set in algorithms
    if args.imb_algorithm is not None:
        model = get_imb_algorithm(args, _net_builder, tb_log, logger)
    else:
        model = get_algorithm(args, _net_builder, tb_log, logger)
    logger.info(f'Number of Trainable Params: {count_parameters(model.model)}')

    # SET Devices for (Distributed) DataParallel
    model.model = send_model_cuda(args, model.model)
    model.ema_model = send_model_cuda(args, model.ema_model, clip_batch=False)
    logger.info(f"Arguments: {model.args}")

    # If args.resume, load checkpoints from args.load_path
    if args.resume and os.path.exists(args.load_path):
        try:
            model.load_model(args.load_path)
        except:
            logger.info("Fail to resume load path {}".format(args.load_path))    
            args.resume = False
    else:
        logger.info("Resume load path {} does not exist".format(args.load_path))

    if hasattr(model, 'warmup'):
        logger.info(("Warmup stage"))
        model.warmup()

    # START TRAINING of FixMatch
    logger.info("Model training")
    model.train()

    # print validation (and test results)
    for key, item in model.results_dict.items():
        logger.info(f"Model result - {key} : {item}")

    if hasattr(model, 'finetune'):
        logger.info("Finetune stage")
        model.finetune()

    logging.warning(f"GPU {args.rank} training is FINISHED")


if __name__ == "__main__":
    args = get_config()
    port = get_port()
    args.dist_url = "tcp://127.0.0.1:" + str(port)
    main(args)


# ablation study:
# config file path: config/run/pivotmatch/pivotmatch61_cifar10_40.yaml

# ablation1: alpha for label propagation
# run23, alpha = 0.7
# change config file's alpha to 0.7, <keep num_pivots = 30>, please check log file for paramter
# python run.py --cfg config/run/pivotmatch/pivotmatch61_cifar10_40.yaml --run_id 23 --gpu_id 0 --seed 0 
# python run.py --cfg config/run/pivotmatch/pivotmatch61_cifar10_40.yaml --run_id 23 --gpu_id 1 --seed 1
# python run.py --cfg config/run/pivotmatch/pivotmatch61_cifar10_40.yaml --run_id 23 --gpu_id 0 --seed 2 

# ablation2: num_pivots 
# run10, num_pivots=10 (I have run seed=0)
# change config file's num_pivots = 10, <keep alpha=0.9>, please check log file for paramter
# python run.py --cfg config/run/pivotmatch/pivotmatch61_cifar10_40.yaml --run_id 10 --gpu_id 1 --seed 1
# python run.py --cfg config/run/pivotmatch/pivotmatch61_cifar10_40.yaml --run_id 10 --gpu_id 0 --seed 2

# run14, num_pivots=100 (I have run seed=0)
# change config file's num_pivots = 100, <keep alpha=0.9>, please check log file for paramter
# python run.py --cfg config/run/pivotmatch/pivotmatch61_cifar10_40.yaml --run_id 14 --gpu_id 1 --seed 1
# python run.py --cfg config/run/pivotmatch/pivotmatch61_cifar10_40.yaml --run_id 14 --gpu_id 0 --seed 2

# run12, num_pivots=50 (I will run seed=0)
# change config file's num_pivots = 50, <keep alpha=0.9>, please check log file for paramter
# python run.py --cfg config/run/pivotmatch/pivotmatch61_cifar10_40.yaml --run_id 12 --gpu_id 1 --seed 1
# python run.py --cfg config/run/pivotmatch/pivotmatch61_cifar10_40.yaml --run_id 12 --gpu_id 0 --seed 2

# run13, num_pivots=70 
# change config file's num_pivots = 70, <keep alpha=0.9>, please check log file for paramter
# python run.py --cfg config/run/pivotmatch/pivotmatch61_cifar10_40.yaml --run_id 13 --gpu_id 1 --seed 0
# python run.py --cfg config/run/pivotmatch/pivotmatch61_cifar10_40.yaml --run_id 13 --gpu_id 0 --seed 1
# python run.py --cfg config/run/pivotmatch/pivotmatch61_cifar10_40.yaml --run_id 13 --gpu_id 1 --seed 2