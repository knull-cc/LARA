import argparse
import torch
import torch.backends
import random
import numpy as np

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.print_args import print_args


def build_parser():
    parser = argparse.ArgumentParser(description='TSF-Lib: a concise time series forecasting framework')

    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name; TSF-Lib only supports long_term_forecast')
    parser.add_argument('--is_training', type=int, default=1, help='status: 1 train+test, 0 test only')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, default='DLinear',
                        help='model name; any file in models/ that defines class Model (e.g. DLinear, iTransformer, PatchTST)')

    # data loader
    parser.add_argument('--data', type=str, default='ETTh1',
                        help='dataset type, options: [ETTh1, ETTh2, ETTm1, ETTm2, custom, PEMS, Solar]')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT-small/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, '
                             'S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s, t, h, d, b, w, m] or e.g. 15min, 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4 (unused here)')
    parser.add_argument('--inverse', action='store_true', default=False, help='inverse output data')

    # model define (shared hyper-parameters; new models read what they need from configs)
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size (num of variates)')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length (PatchTST-style models)')
    parser.add_argument('--stride', type=int, default=8, help='stride (PatchTST-style models)')
    parser.add_argument('--distil', action='store_false', default=True,
                        help='whether to use distilling in encoder (used in setting name)')
    parser.add_argument('--individual', action='store_true', default=False,
                        help='DLinear: a linear layer for each variate(channel) individually')
    parser.add_argument('--cycle', type=int, default=24,
                        help='cycle length for GTR-style global temporal retrieval')
    parser.add_argument('--use_revin', type=int, default=1,
                        help='GTR: 1 enables RevIN normalization, 0 disables it')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='Exp', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', default=False, help='use automatic mixed precision training')

    # GPU
    parser.add_argument('--use_gpu', action='store_true', default=True, help='use gpu (default: on)')
    parser.add_argument('--no_use_gpu', action='store_false', dest='use_gpu', help='disable gpu (force cpu)')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type: cuda or mps')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False, help='use multiple gpus')
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multiple gpus')

    # metrics
    parser.add_argument('--use_dtw', action='store_true', default=False,
                        help='enable dtw metric (time consuming; default: off)')

    # data augmentation (off by default)
    parser.add_argument('--augmentation_ratio', type=int, default=0, help='how many times to augment')
    parser.add_argument('--seed', type=int, default=2, help='randomization seed for augmentation')

    # LARA MVP adapter
    parser.add_argument('--lara_top_m', type=int, default=32, help='LARA top-M retrieval candidates')
    parser.add_argument('--lara_retrieval_chunk', type=int, default=4096, help='LARA memory search chunk size')
    parser.add_argument('--lara_temperature', type=float, default=0.1, help='LARA retrieval aggregation temperature')
    parser.add_argument('--lara_rank_temperature', type=float, default=0.1, help='LARA utility ranking temperature')
    parser.add_argument('--lara_lambda_rank', type=float, default=0.3, help='LARA rank loss weight')
    parser.add_argument('--lara_lambda_pair', type=float, default=0.0,
                        help='LARA pairwise utility margin loss weight')
    parser.add_argument('--lara_pair_margin', type=float, default=0.2,
                        help='LARA pairwise utility margin between best and worst candidate scores')
    parser.add_argument('--lara_lambda_score', type=float, default=0.0,
                        help='LARA score-to-utility alignment loss weight')
    parser.add_argument('--lara_score_loss', type=str, default='mse', choices=['mse', 'corr'],
                        help='LARA score alignment loss type')
    parser.add_argument('--lara_lambda_sparse', type=float, default=0.01, help='LARA sparse entropy loss weight')
    parser.add_argument('--lara_lambda_gate', type=float, default=0.0, help='LARA supervised gate loss weight')
    parser.add_argument('--lara_sparse_mode', type=str, default='softmax', choices=['softmax', 'sparsemax'],
                        help='LARA candidate weighting function')
    parser.add_argument('--lara_key_mode', type=str, default='auto', choices=['auto', 'flatten', 'mean'],
                        help='LARA memory key mode')
    parser.add_argument('--lara_max_key_dim', type=int, default=8192,
                        help='LARA auto key mode uses flattened keys only under this dimension')
    parser.add_argument('--lara_overlap_margin', type=int, default=0,
                        help='extra training overlap mask margin in samples')
    parser.add_argument('--lara_gate', type=str, default='scalar', choices=['scalar', 'horizon'],
                        help='LARA fusion gate type')
    parser.add_argument('--lara_gate_target', type=str, default='alpha', choices=['alpha', 'oracle'],
                        help='LARA supervised gate target: alpha uses weighted interpolation labels; oracle uses true host-vs-retrieval benefit')
    parser.add_argument('--lara_host_ckpt', type=str, default='',
                        help='optional host checkpoint loaded into the LARA host')
    parser.add_argument('--lara_freeze_host', action='store_true', default=False,
                        help='freeze the host forecaster and train only the LARA adapter')
    parser.add_argument('--lara_alpha_step', type=float, default=0.1,
                        help='alpha grid step for oracle utility labels')
    parser.add_argument('--lara_offset_align', action='store_true', default=True,
                        help='align candidate futures by query_last + (candidate_future - candidate_past_last)')
    parser.add_argument('--no_lara_offset_align', action='store_false', dest='lara_offset_align',
                        help='disable LARA offset-aligned candidate futures')
    parser.add_argument('--lara_oracle_ms', type=str, default='1,3,5,10,20,50',
                        help='comma-separated M values for LARA oracle candidate top-M curves')
    parser.add_argument('--lara_oracle_topk', type=str, default='1,3,5',
                        help='comma-separated k values for LARA oracle true-error top-k average aggregation')

    # reproducibility
    parser.add_argument('--random_seed', type=int, default=2021, help='global random seed')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # reproducibility
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # device
    if torch.cuda.is_available() and args.use_gpu:
        args.gpu_type = 'cuda'
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU: cuda:{}'.format(args.gpu))
    elif args.use_gpu and getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        args.gpu_type = 'mps'
        args.device = torch.device('mps')
        print('Using GPU: mps')
    else:
        args.use_gpu = False
        args.device = torch.device('cpu')
        print('Using CPU')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    assert args.task_name == 'long_term_forecast', \
        'TSF-Lib only supports task_name=long_term_forecast'

    print('Args in experiment:')
    print_args(args)

    Exp = Exp_Long_Term_Forecast

    def make_setting(ii):
        return '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
            args.task_name, args.model_id, args.model, args.data, args.features,
            args.seq_len, args.label_len, args.pred_len, args.d_model, args.n_heads,
            args.e_layers, args.d_layers, args.d_ff, args.factor, args.embed, args.des, ii)

    def empty_cache():
        if args.use_gpu and args.gpu_type == 'mps':
            torch.backends.mps.empty_cache()
        elif args.use_gpu and args.gpu_type == 'cuda':
            torch.cuda.empty_cache()

    if args.is_training:
        for ii in range(args.itr):
            exp = Exp(args)
            setting = make_setting(ii)
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            empty_cache()
    else:
        exp = Exp(args)
        setting = make_setting(0)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        empty_cache()


if __name__ == '__main__':
    main()
