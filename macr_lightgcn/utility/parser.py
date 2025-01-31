'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run NGCF.")
    parser.add_argument('--weights_path', nargs='?', default='',
                        help='Store model path.')
    parser.add_argument('--data_path', nargs='?', default='../data/',
                        help='Input data path.')
    parser.add_argument('--proj_path', nargs='?', default='',
                        help='Project path.')

    parser.add_argument('--dataset', nargs='?', default='gowalla',
                        help='Choose a dataset')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='0: No pretrain, -1: Pretrain with the learned embeddings, 1:Pretrain with stored models.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--is_norm', type=int, default=1,
                    help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=1000,
                        help='Number of epoch.')

    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--layer_size', nargs='?', default='[]',
                        help='Output sizes of every layer')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')

    parser.add_argument('--regs', nargs='?', default='[1e-5,1e-5,1e-2]',
                        help='Regularizations.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--c', type=float, default=-1,
                        help='Value of C. -1 means automatic selection')

    parser.add_argument('--model_type', nargs='?', default='lightgcn',
                        help='Specify the name of model (lightgcn).')
    parser.add_argument('--adj_type', nargs='?', default='pre',
                        help='Specify the type of the adjacency (laplacian) matrix from {plain, norm, mean}.')
    parser.add_argument('--alg_type', nargs='?', default='lightgcn',
                        help='Specify the type of the graph convolutional layer from {ngcf, gcn, gcmc}.')

    parser.add_argument('--gpu_id', type=int, default=0,
                        help='0 for NAIS_prod, 1 for NAIS_concat')

    parser.add_argument('--node_dropout_flag', type=int, default=0,
                        help='0: Disable node dropout, 1: Activate node dropout')
    parser.add_argument('--node_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')

    parser.add_argument('--Ks', nargs='?', default='[1,5,10,15,20,30]',
                        help='Top k(s) recommend')

    parser.add_argument('--save_flag', type=int, default=1,
                        help='0: Disable model saver, 1: Activate model saver')

    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')


    parser.add_argument('--saveID', nargs='?', default='',
                        help='Specify model save path.')
    parser.add_argument('--base', type=float, default=-1.,
                        help='check range base.')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='log\'s interval while training')
    parser.add_argument('--only_test', type = int, default = 0,
                        help = 'need train or not.') 




    parser.add_argument('--loss', nargs='?', default='bpr',
                        help='bpr/bce.')
    parser.add_argument('--alpha', type=float, default=1e-3,
                        help='alpha')
    parser.add_argument('--beta', type=float, default=1e-3,
                        help='alpha')
    parser.add_argument('--test', nargs='?', default='normal',
                        help='test:normal | rubiboth')
    parser.add_argument('--early_stop', type = int, default = 1,
                        help = 'early_stop') 
    parser.add_argument('--start', type=float, default=-1.,
                        help='check c start.')
    parser.add_argument('--end', type=float, default=1.,
                        help='check c end.')
    parser.add_argument('--step', type=int, default=20,
                        help='check c step.')   

    parser.add_argument('--out', type=int, default=0) 


    #new args
    parser.add_argument('--neg_sample', type=int, default=32,
                        help='negative sample ratio.')    
    parser.add_argument('--tau', type=float, default=0.2,
                        help='temperature parameter for L1')
    parser.add_argument('--tau_info', type=float, default=0.1,
                        help='temperature parameter for L2')
    parser.add_argument('--w_lambda', type=float, default=0.5,
                        help='weight for combining l1 and l2.')
    parser.add_argument('--warm_up', type=int, default=5,
                        help='warm up epochs for initial tau')
    parser.add_argument('--tau_decay',type=float,default=1,
                        help='the decay rate for tau, default 1 for no decay')
    parser.add_argument('--tau_cut', type=float, default=0.05,
                        help='min tau after decay')
    parser.add_argument('--w_lambda_decay',type=float,default=1,
                        help='the decay rate for w_lambda, default 1 for no decay')
    parser.add_argument('--lambda_cut',type=float,default=0.05,
                        help='min w_lambda after decay')
    parser.add_argument('--freeze',type=int,default=0)
    
    parser.add_argument('--freeze_epoch',type=int,default=10)

    parser.add_argument('--pop_branch',type=str,default="lightgcn")

    parser.add_argument('--pop_reduct',type=int, default=0)
    
    parser.add_argument('--inbatch_sample',type=int,default=1)

    parser.add_argument('--cf_pen', type=float, default=0.1,
                        help='Imbalance loss.')
    return parser.parse_args()
