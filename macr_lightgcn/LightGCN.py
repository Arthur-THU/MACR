'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)

version:
Parallelized sampling on CPU
C++ evaluation for top-k recommendation
'''


import os
import sys
import threading
import tensorflow as tf
from util import DataIterator
from tensorflow.python.client import device_lib
from tqdm import tqdm
from utility.helper import *
from utility.batch_test import *
from utility.visualization import plot_tsne_embed
from evaluator import ProxyEvaluator
import scipy.sparse as sp

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

gpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
cpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'CPU']

class LightGCN(object):
    def __init__(self, data_config, pretrain_data,user_pop_num=-1,item_pop_num=-1,user_pop=None,item_pop=None,pop_branch="lightgcn",pop_reduct=0,p_matrix=None):
        # argument settings
        self.model_type = 'LightGCN'
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type
        self.pretrain_data = pretrain_data
        self.pop_branch=pop_branch
        self.pop_reduct=pop_reduct
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_fold = 100
        self.norm_adj = data_config['norm_adj']
        self.norm_adj_pop = data_config['norm_adj_pop']
        self.n_nonzero_elems = self.norm_adj.count_nonzero()
        self.lr = args.lr
        self.cf_pen = args.cf_pen
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        self.log_dir=self.create_model_str()
        self.verbose = args.verbose
        self.Ks = eval(args.Ks)
        self.user_pop_idx=tf.constant([user_pop[i] for i in range(self.n_users)])
        self.item_pop_idx=tf.constant([item_pop[i] for i in range(self.n_items)])
        

        

        self.tau = args.tau
        self.temp = args.tau_info
        self.w_lambda = args.w_lambda
        if args.loss=="mf_info" or args.loss=="dyninfo":
            if not args.inbatch_sample:
                self.neg_sample=args.neg_sample
            else:
                self.neg_sample=args.batch_size-1
        else:
            self.neg_sample=args.neg_sample
        self.pop_partition_user=user_pop_num
        self.pop_partition_item=item_pop_num

        '''
        *********************************************************
        Create Placeholder for Input Data & Dropout.
        '''
        self.p = tf.constant(value = p_matrix, dtype = tf.float32)
        # placeholder definition
        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))
        self.items = tf.placeholder(tf.int32, shape=(None,))
        self.ctrl_items = tf.placeholder(tf.int32, shape=(None,))
        self.pos_item_p = tf.nn.embedding_lookup(self.p, self.pos_items)
        self.neg_item_p = tf.nn.embedding_lookup(self.p, self.neg_items)

        self.users_pop = tf.placeholder(tf.int32, shape = (None,))
        self.pos_items_pop = tf.placeholder(tf.int32, shape = (None,))
        self.neg_items_pop = tf.placeholder(tf.int32, shape = (None,))

        
        
        self.node_dropout_flag = args.node_dropout_flag
        self.node_dropout = tf.placeholder(tf.float32, shape=[None])
        self.mess_dropout = tf.placeholder(tf.float32, shape=[None])
        with tf.name_scope('TRAIN_LOSS'):
            self.train_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('train_loss', self.train_loss)
            self.train_mf_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('train_mf_loss', self.train_mf_loss)
            self.train_emb_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('train_emb_loss', self.train_emb_loss)
            self.train_reg_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('train_reg_loss', self.train_reg_loss)
        self.merged_train_loss = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, 'TRAIN_LOSS'))
        
        
        with tf.name_scope('TRAIN_ACC'):
            self.train_rec_first = tf.placeholder(tf.float32)
            #record for top(Ks[0])
            tf.summary.scalar('train_rec_first', self.train_rec_first)
            self.train_rec_last = tf.placeholder(tf.float32)
            #record for top(Ks[-1])
            tf.summary.scalar('train_rec_last', self.train_rec_last)
            self.train_ndcg_first = tf.placeholder(tf.float32)
            tf.summary.scalar('train_ndcg_first', self.train_ndcg_first)
            self.train_ndcg_last = tf.placeholder(tf.float32)
            tf.summary.scalar('train_ndcg_last', self.train_ndcg_last)
        self.merged_train_acc = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, 'TRAIN_ACC'))

        with tf.name_scope('TEST_LOSS'):
            self.test_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('test_loss', self.test_loss)
            self.test_mf_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('test_mf_loss', self.test_mf_loss)
            self.test_emb_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('test_emb_loss', self.test_emb_loss)
            self.test_reg_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('test_reg_loss', self.test_reg_loss)
        self.merged_test_loss = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, 'TEST_LOSS'))

        with tf.name_scope('TEST_ACC'):
            self.test_rec_first = tf.placeholder(tf.float32)
            tf.summary.scalar('test_rec_first', self.test_rec_first)
            self.test_rec_last = tf.placeholder(tf.float32)
            tf.summary.scalar('test_rec_last', self.test_rec_last)
            self.test_ndcg_first = tf.placeholder(tf.float32)
            tf.summary.scalar('test_ndcg_first', self.test_ndcg_first)
            self.test_ndcg_last = tf.placeholder(tf.float32)
            tf.summary.scalar('test_ndcg_last', self.test_ndcg_last)
        self.merged_test_acc = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, 'TEST_ACC'))
        """
        *********************************************************
        Create Model Parameters (i.e., Initialize Weights).
        """
        # initialization of model parameters
        self.weights = self._init_weights()

        """
        *********************************************************
        Compute Graph-based Representations of all users & items via Message-Passing Mechanism of Graph Neural Networks.
        Different Convolutional Layers:
            1. ngcf: defined in 'Neural Graph Collaborative Filtering', SIGIR2019;
            2. gcn:  defined in 'Semi-Supervised Classification with Graph Convolutional Networks', ICLR2018;
            3. gcmc: defined in 'Graph Convolutional Matrix Completion', KDD2018;
        """
        if self.alg_type in ['lightgcn']:
            self.ua_embeddings, self.ia_embeddings = self._create_lightgcn_embed()
            self.ua_pop_embeddings,self.ia_pop_embeddings = self._create_lightgcn_embed(is_pop=True)
        
        

            
        elif self.alg_type in ['ngcf']:
            self.ua_embeddings, self.ia_embeddings = self._create_ngcf_embed()

        elif self.alg_type in ['gcn']:
            self.ua_embeddings, self.ia_embeddings = self._create_gcn_embed()

        elif self.alg_type in ['gcmc']:
            self.ua_embeddings, self.ia_embeddings = self._create_gcmc_embed()

        """
        *********************************************************
        Establish the final representations for user-item pairs in batch.
        """

        self.caus_e_i = tf.concat([self.ia_embeddings,self.weights['ctrl_embedding']],axis=0)
        self.u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.users)
        self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.caus_e_i, self.pos_items)
        self.neg_i_g_embeddings = tf.nn.embedding_lookup(self.caus_e_i, self.neg_items)
        self.ctrl_i_g_embeddings = tf.nn.embedding_lookup(self.caus_e_i, self.ctrl_items)
        self.all_i_g_embeddings = tf.nn.embedding_lookup(self.caus_e_i, self.items)
        self.u_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['user_embedding'], self.users)
        self.pos_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['item_embedding'], self.pos_items)
        self.neg_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['item_embedding'], self.neg_items)



        
        if self.pop_branch=="mf":
            self.user_pop_embedding = tf.nn.embedding_lookup(self.weights['tiled_user_pop_embedding'], self.users)
            self.pos_item_pop_embedding = tf.nn.embedding_lookup(self.weights['tiled_item_pop_embedding'], self.pos_items)
            self.neg_item_pop_embedding = tf.nn.embedding_lookup(self.weights['tiled_item_pop_embedding'], self.neg_items)
        elif self.pop_branch=="lightgcn":
            if self.pop_reduct==0:
                self.user_pop_embedding = tf.nn.embedding_lookup(self.ua_pop_embeddings, self.users)
                self.pos_item_pop_embedding = tf.nn.embedding_lookup(self.ia_pop_embeddings, self.pos_items)
                self.neg_item_pop_embedding = tf.nn.embedding_lookup(self.ia_pop_embeddings, self.neg_items)
            else:
                self.user_pop_embedding = tf.nn.embedding_lookup(self.ua_pop_embeddings, self.users_pop)
                self.pos_item_pop_embedding = tf.nn.embedding_lookup(self.ia_pop_embeddings, self.pos_items_pop)
                self.neg_item_pop_embedding = tf.nn.embedding_lookup(self.ia_pop_embeddings, self.neg_items_pop)

        """
        *********************************************************
        Establish 2 brach.
        """
        self.alpha = args.alpha
        self.beta = args.beta
        self.rubi_c = tf.Variable(tf.zeros([1]), name = 'rubi_c')
        self.sigmoid_yu = tf.squeeze(tf.nn.sigmoid(tf.matmul(self.weights['user_embedding'], self.w_user)))
        self.sigmoid_yi = tf.squeeze(tf.nn.sigmoid(tf.matmul(self.weights['item_embedding'], self.w)))
        """
        *********************************************************
        Inference for the testing phase.
        """
        self.constant_e = self.weights['constant_embedding']
        self.batch_ratings = tf.matmul(self.u_g_embeddings, self.pos_i_g_embeddings, transpose_a=False, transpose_b=True)
        self.batch_ratings_pop = tf.matmul(self.user_pop_embedding, self.pos_item_pop_embedding, transpose_a=False, transpose_b=True)
        self.batch_ratings_constant = tf.matmul(self.constant_e, self.pos_i_g_embeddings, transpose_a=False, transpose_b=True)
        self.batch_ratings_causal_c = self.batch_ratings - self.batch_ratings_constant
        """
        *********************************************************
        Generate Predictions & Optimize via BPR loss.
        """
        self.mf_loss_info_1, self.mf_loss_info_2, self.reg_loss_info, self.reg_loss_freeze, self.reg_loss_norm, self.loss_mf_ori, self.debug = self.create_dyninfonce_loss(self.u_g_embeddings,
                                                                          self.pos_i_g_embeddings,
                                                                          self.neg_i_g_embeddings,
                                                                          self.user_pop_embedding,
                                                                          self.pos_item_pop_embedding,
                                                                          self.neg_item_pop_embedding)


        self.loss_info = self.mf_loss_info_1 + self.mf_loss_info_2 + self.reg_loss_info
        self.loss_mf_info = self.reg_loss_norm + self.loss_mf_ori
        self.opt_info = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_info)
        
        trainable_v1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        #print(trainable_v1)
        pop_list=[self.weights["user_pop_embedding"], self.weights["item_pop_embedding"]]
        norm_list=[i for i in trainable_v1 if i not in pop_list]

        self.loss_freeze=self.reg_loss_freeze + self.mf_loss_info_2
        self.opt_freeze = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_freeze,var_list=pop_list)
        self.opt_none_freeze = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_info,var_list=norm_list)

        self.opt_mf_info = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_mf_info)


        self.cause_mf_loss, self.cause_reg_loss, self.cause_cf_loss = self.create_cause_loss(self.u_g_embeddings, self.pos_i_g_embeddings, self.neg_i_g_embeddings,self.all_i_g_embeddings, self.ctrl_i_g_embeddings)
        self.loss_cause = self.cause_mf_loss + self.cause_reg_loss + self.cause_cf_loss

        self.opt_cause = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_cause)


        self.mf_loss, self.emb_loss, self.reg_loss = self.create_bpr_loss(self.u_g_embeddings,
                                                                          self.pos_i_g_embeddings,
                                                                          self.neg_i_g_embeddings)
        self.loss = self.mf_loss + self.emb_loss

        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)



        self.mf_loss_bce, self.emb_loss_bce, self.reg_loss_bce = self.create_bce_loss(self.u_g_embeddings,
                                                                          self.pos_i_g_embeddings,
                                                                          self.neg_i_g_embeddings)
        self.loss_bce = self.mf_loss_bce + self.emb_loss_bce
        self.opt_bce = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_bce)



        self.mf_loss_two_bce1, self.emb_loss_two_bce1, self.reg_loss_two_bce1 = self.create_bce_loss_two_brach1(self.u_g_embeddings,
                                                                          self.pos_i_g_embeddings,
                                                                          self.neg_i_g_embeddings)
        self.loss_two_bce1 = self.mf_loss_two_bce1 + self.emb_loss_two_bce1
        self.opt_two_bce1 = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_two_bce1)


        self.mf_loss_two_bce_both, self.emb_loss_two_bce_both, self.reg_loss_two_bce_both = self.create_bce_loss_two_brach_both(self.u_g_embeddings,
                                                                          self.pos_i_g_embeddings,
                                                                          self.neg_i_g_embeddings)
        self.loss_two_bce_both = self.mf_loss_two_bce_both + self.emb_loss_two_bce_both
        self.opt_two_bce_both = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_two_bce_both)



        self.mf_loss_two_bce2, self.emb_loss_two_bce2, self.reg_loss_two_bce2 = self.create_bce_loss_two_brach2(self.u_g_embeddings,
                                                                          self.pos_i_g_embeddings,
                                                                          self.neg_i_g_embeddings)
        self.loss_two_bce2 = self.mf_loss_two_bce2 + self.emb_loss_two_bce2
        self.opt_two_bce2 = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_two_bce2)


        self.ipw_mf_loss, self.ipw_reg_loss = self.create_ipw_loss(self.u_g_embeddings, self.pos_i_g_embeddings, self.neg_i_g_embeddings, self.pos_item_p, self.neg_item_p)

        self.ipw_loss = self.ipw_mf_loss + self.ipw_reg_loss

        self.opt_ipw = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.ipw_loss)
    
    
    def create_model_str(self):
        log_dir = '/' + self.alg_type+'/layers_'+str(self.n_layers)+'/dim_'+str(self.emb_dim)
        log_dir+='/'+args.dataset+'/lr_' + str(self.lr) + '/reg_' + str(self.decay)
        return log_dir

    def update_c(self, sess, c):
        sess.run(tf.assign(self.constant_e, c*tf.ones([1, self.emb_dim])))

    def _init_weights(self):
        all_weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()
        #initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None)
        all_weights["constant_embedding"] = tf.Variable(tf.ones([1, self.emb_dim]), name='constant_embedding')
        if self.pretrain_data is None:
            all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embedding')
            all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name='item_embedding')
            all_weights['ctrl_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name='ctrl_embedding')
            print('using xavier initialization')
        else:
            all_weights['user_embedding'] = tf.Variable(initial_value=self.pretrain_data['user_embed'], trainable=True,
                                                        name='user_embedding', dtype=tf.float32)
            all_weights['item_embedding'] = tf.Variable(initial_value=self.pretrain_data['item_embed'], trainable=True,
                                                        name='item_embedding', dtype=tf.float32)
            print('using pretrained initialization')

        if self.pop_partition_user!=-1:
            all_weights["user_pop_embedding"] = tf.Variable(initializer([self.pop_partition_user, self.emb_dim]), name = 'user_pop_embedding')
            all_weights["item_pop_embedding"] = tf.Variable(initializer([self.pop_partition_item, self.emb_dim]), name = 'item_pop_embedding')
            all_weights["tiled_user_pop_embedding"] = tf.nn.embedding_lookup(all_weights["user_pop_embedding"],self.user_pop_idx)
            all_weights["tiled_item_pop_embedding"] = tf.nn.embedding_lookup(all_weights["item_pop_embedding"],self.item_pop_idx)

        self.w = tf.Variable(initializer([self.emb_dim, 1]), name = 'item_branch')
        self.w_user = tf.Variable(initializer([self.emb_dim, 1]), name = 'user_branch')
            
        self.weight_size_list = [self.emb_dim] + self.weight_size
        
        for k in range(self.n_layers):
            all_weights['W_gc_%d' %k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_gc_%d' % k)
            all_weights['b_gc_%d' %k] = tf.Variable(
                initializer([1, self.weight_size_list[k+1]]), name='b_gc_%d' % k)

            all_weights['W_bi_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_bi_%d' % k)
            all_weights['b_bi_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_bi_%d' % k)

            all_weights['W_mlp_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_mlp_%d' % k)
            all_weights['b_mlp_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k+1]]), name='b_mlp_%d' % k)

        return all_weights
    def _split_A_hat(self, X, is_pop=False):
        A_fold_hat = []
        if is_pop==False:
            fold_len = (self.n_users + self.n_items) // self.n_fold
            for i_fold in range(self.n_fold):
                start = i_fold * fold_len
                if i_fold == self.n_fold -1:
                    end = self.n_users + self.n_items
                else:
                    end = (i_fold + 1) * fold_len

                A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        else:
            fold_len = (self.pop_partition_user + self.pop_partition_item) // self.n_fold
            for i_fold in range(self.n_fold):
                start = i_fold * fold_len
                if i_fold == self.n_fold -1:
                    end = self.pop_partition_user + self.pop_partition_item
                else:
                    end = (i_fold + 1) * fold_len

                A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
            
        return A_fold_hat

    def _split_A_hat_node_dropout(self, X, is_pop=False):
        A_fold_hat = []

        if is_pop==False:
            fold_len = (self.n_users + self.n_items) // self.n_fold
            for i_fold in range(self.n_fold):
                start = i_fold * fold_len
                if i_fold == self.n_fold -1:
                    end = self.n_users + self.n_items
                else:
                    end = (i_fold + 1) * fold_len

                temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
                n_nonzero_temp = X[start:end].count_nonzero()
                A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))
        else:
            fold_len = (self.pop_partition_user + self.pop_partition_item) // self.n_fold
            for i_fold in range(self.n_fold):
                start = i_fold * fold_len
                if i_fold == self.n_fold -1:
                    end = self.pop_partition_user + self.pop_partition_item
                else:
                    end = (i_fold + 1) * fold_len

                temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
                n_nonzero_temp = X[start:end].count_nonzero()
                A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))

        return A_fold_hat

    def _create_lightgcn_embed(self,is_pop=False):
        #print(self.norm_adj.shape)
        
        
        
        if is_pop==False:
            if self.node_dropout_flag:
                A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
            else:
                A_fold_hat = self._split_A_hat(self.norm_adj)
            ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)
        else:
            if self.pop_reduct:
                if self.node_dropout_flag:
                    A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj_pop,is_pop=True)
                else:
                    A_fold_hat = self._split_A_hat(self.norm_adj_pop,is_pop=True)
                ego_embeddings = tf.concat([self.weights['user_pop_embedding'], self.weights['item_pop_embedding']], axis=0)
            else:
                if self.node_dropout_flag:
                    A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
                else:
                    A_fold_hat = self._split_A_hat(self.norm_adj)
                ego_embeddings = tf.concat([self.weights['tiled_user_pop_embedding'], self.weights['tiled_item_pop_embedding']], axis=0)


        all_embeddings = [ego_embeddings]
        
        for k in range(0, self.n_layers):

            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            side_embeddings = tf.concat(temp_embed, 0)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings=tf.stack(all_embeddings,1)
        all_embeddings=tf.reduce_mean(all_embeddings,axis=1,keepdims=False)
        if is_pop==False or self.pop_reduct==0:
            u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        else:
            u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.pop_partition_user, self.pop_partition_item], 0)
        return u_g_embeddings, i_g_embeddings
    
    def _create_ngcf_embed(self):
        if self.node_dropout_flag:
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj)

        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):

            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            side_embeddings = tf.concat(temp_embed, 0)
            sum_embeddings = tf.nn.leaky_relu(tf.matmul(side_embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])



            # bi messages of neighbors.
            bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)
            # transformed bi messages of neighbors.
            bi_embeddings = tf.nn.leaky_relu(tf.matmul(bi_embeddings, self.weights['W_bi_%d' % k]) + self.weights['b_bi_%d' % k])
            # non-linear activation.
            ego_embeddings = sum_embeddings + bi_embeddings

            # message dropout.
            ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - self.mess_dropout[k])

            # normalize the distribution of embeddings.
            norm_embeddings = tf.nn.l2_normalize(ego_embeddings, axis=1)

            all_embeddings += [norm_embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings
    
    
    def _create_gcn_embed(self):
        A_fold_hat = self._split_A_hat(self.norm_adj)
        embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)


        all_embeddings = [embeddings]

        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings))

            embeddings = tf.concat(temp_embed, 0)
            embeddings = tf.nn.leaky_relu(tf.matmul(embeddings, self.weights['W_gc_%d' %k]) + self.weights['b_gc_%d' %k])
            embeddings = tf.nn.dropout(embeddings, 1 - self.mess_dropout[k])

            all_embeddings += [embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings
    
    def _create_gcmc_embed(self):
        A_fold_hat = self._split_A_hat(self.norm_adj)

        embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings = []

        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings))
            embeddings = tf.concat(temp_embed, 0)
            # convolutional layer.
            embeddings = tf.nn.leaky_relu(tf.matmul(embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])
            # dense layer.
            mlp_embeddings = tf.matmul(embeddings, self.weights['W_mlp_%d' %k]) + self.weights['b_mlp_%d' %k]
            mlp_embeddings = tf.nn.dropout(mlp_embeddings, 1 - self.mess_dropout[k])

            all_embeddings += [mlp_embeddings]
        all_embeddings = tf.concat(all_embeddings, 1)

        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    def create_ipw_loss(self, users, pos_items, neg_items, pos_item_p, neg_item_p):
        pos_scores = tf.sigmoid(tf.reduce_sum(tf.multiply(users, pos_items), axis=1))   #users, pos_items, neg_items have the same shape
        neg_scores = tf.sigmoid(tf.reduce_sum(tf.multiply(users, neg_items), axis=1))
        
        self.temp2 = pos_item_p
        self.temp3 = tf.divide(pos_scores, pos_item_p)

        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer/self.batch_size

        # maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))

        weighted_pos_item_score=tf.multiply(pos_scores, tf.sigmoid(1/pos_item_p))*10
        weighted_neg_item_score=tf.multiply(neg_scores, tf.sigmoid(1/neg_item_p))*10

        pos_item_score_exp=tf.exp(weighted_pos_item_score)
        neg_item_score_exp=tf.exp(weighted_neg_item_score)

        mf_loss=tf.reduce_mean(tf.negative(tf.log(pos_item_score_exp/(pos_item_score_exp+neg_item_score_exp))))
        reg_loss = self.decay * regularizer
        return mf_loss, reg_loss


    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)   #users, pos_items, neg_items have the same shape
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        
        regularizer = tf.nn.l2_loss(self.u_g_embeddings_pre) + tf.nn.l2_loss(
                self.pos_i_g_embeddings_pre) + tf.nn.l2_loss(self.neg_i_g_embeddings_pre)
        regularizer = regularizer / self.batch_size
        
        mf_loss = tf.negative(tf.reduce_mean(tf.log(1e-9+tf.nn.sigmoid(pos_scores - neg_scores))))
        # mf_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))
        
        emb_loss = self.decay * regularizer

        reg_loss = tf.constant(0.0, tf.float32, [1])

        return mf_loss, emb_loss, reg_loss

    
    def create_cause_loss(self, users, pos_items, neg_items, item_embed, control_embed):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)   #users, pos_items, neg_items have the same shape
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)

        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer/self.batch_size

        maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores)+1e-10)

        mf_loss = tf.negative(tf.reduce_mean(maxi))
        reg_loss = self.decay * regularizer

        #counter factual loss

        #cf_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(tf.subtract(item_embed, control_embed)), axis=1))
        cf_loss = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(tf.nn.l2_normalize(item_embed,axis=0), tf.nn.l2_normalize(control_embed,axis=0)))))
        cf_loss = cf_loss * self.cf_pen #/ self.batch_size

        return mf_loss, reg_loss, cf_loss

    def create_bce_loss(self, users, pos_items, neg_items):
        pos_scores = tf.nn.sigmoid(tf.reduce_sum(tf.multiply(users, pos_items), axis=1))   #users, pos_items, neg_items have the same shape
        neg_scores = tf.nn.sigmoid(tf.reduce_sum(tf.multiply(users, neg_items), axis=1))

        regularizer = tf.nn.l2_loss(self.u_g_embeddings_pre) + tf.nn.l2_loss(
                self.pos_i_g_embeddings_pre) + tf.nn.l2_loss(self.neg_i_g_embeddings_pre)
        regularizer = regularizer/self.batch_size

        mf_loss = tf.reduce_mean(tf.negative(tf.log(pos_scores+1e-9))+tf.negative(tf.log(1-neg_scores+1e-9)))
        
        emb_loss = self.decay * regularizer

        reg_loss = tf.constant(0.0, tf.float32, [1])
        
        return mf_loss, emb_loss, reg_loss

    def create_dyninfonce_loss(self, users, pos_items, neg_items, users_pop, pos_items_pop, neg_items_pop):

        tiled_usr=tf.reshape(tf.tile(users,[1,self.neg_sample]),[-1,self.emb_dim])
        tiled_usr_pop=tf.reshape(tf.tile(users_pop,[1,self.neg_sample]),[-1,self.emb_dim])


        user_n2=tf.norm(users,ord=2,axis=1)
        user_pop_n2=tf.norm(users_pop,ord=2,axis=1)
        tiled_usr_n2=tf.norm(tiled_usr,ord=2,axis=1)
        tiled_usr_pop_n2=tf.norm(tiled_usr_pop,ord=2,axis=1)#tf.sqrt(tf.reduce_sum(tf.multiply(tiled_usr_pop,tiled_usr_pop),axis=1))
        pos_item_n2=tf.norm(pos_items,ord=2,axis=1)
        neg_item_n2=tf.norm(neg_items,ord=2,axis=1)
        neg_item_pop_n2=tf.norm(neg_items_pop,ord=2,axis=1)
        pos_item_pop_n2=tf.norm(pos_items_pop,ord=2,axis=1)

        pos_item_pop_prod=tf.reduce_sum(tf.multiply(users_pop,pos_items_pop),axis=1)
        neg_item_pop_prod=tf.reduce_sum(tf.multiply(tiled_usr_pop,neg_items_pop),axis=1)
        pos_item_prod=tf.reduce_sum(tf.multiply(users,pos_items),axis=1)
        neg_item_prod=tf.reduce_sum(tf.multiply(tiled_usr,neg_items),axis=1)
        #pos_item_score=tf.sigmoid(pos_item_prod)
        #neg_item_score=tf.sigmoid(neg_item_prod)
        # pos_item_pop_score=tf.sigmoid(pos_item_pop_prod)/self.temp
        # neg_item_pop_score=tf.sigmoid(neg_item_pop_prod)/self.temp


        pos_item_score=pos_item_prod/user_n2/pos_item_n2
        neg_item_score=neg_item_prod/tiled_usr_n2/neg_item_n2
        pos_item_pop_score=pos_item_pop_prod/user_pop_n2/pos_item_pop_n2/self.temp
        neg_item_pop_score=neg_item_pop_prod/tiled_usr_pop_n2/neg_item_pop_n2/self.temp

        pos_item_score_mf_exp=tf.exp(pos_item_score/self.tau)
        neg_item_score_mf_exp=tf.reduce_sum(tf.exp(tf.reshape(neg_item_score/self.tau,[-1,self.neg_sample])),axis=1)
        loss_mf=tf.reduce_mean(tf.negative(tf.log(pos_item_score_mf_exp/(pos_item_score_mf_exp+neg_item_score_mf_exp))))


        neg_item_pop_score_exp=tf.reduce_sum(tf.exp(tf.reshape(neg_item_pop_score,[-1,self.neg_sample])),axis=1)
        pos_item_pop_score_exp=tf.exp(pos_item_pop_score)
        loss2=self.w_lambda*tf.reduce_mean(tf.negative(tf.log(pos_item_pop_score_exp/(pos_item_pop_score_exp+neg_item_pop_score_exp))))

        debug=tf.sigmoid(pos_item_pop_prod)

        weighted_pos_item_score=tf.multiply(pos_item_score,tf.sigmoid(pos_item_pop_prod))/self.tau
        weighted_neg_item_score=tf.multiply(neg_item_score,tf.sigmoid(neg_item_pop_prod))/self.tau

        self.ensemble_pos_score=weighted_pos_item_score
        self.ensemble_neg_score=weighted_neg_item_score

        self.pop_pos_score=pos_item_pop_score
        self.pop_neg_score=neg_item_pop_score

        self.pos_score=pos_item_score/self.tau
        self.neg_score=neg_item_score/self.tau

        self.ensemble_pos_score_ori=tf.multiply(pos_item_prod,tf.sigmoid(pos_item_pop_score))
        self.ensemble_neg_score_ori=tf.multiply(neg_item_prod,tf.sigmoid(neg_item_pop_score))

        self.pop_pos_score_ori=pos_item_pop_prod
        self.pop_neg_score_ori=neg_item_pop_prod

        self.pos_score_ori=pos_item_prod
        self.neg_score_ori=neg_item_prod

        #weighted_pos_item_score=tf.multiply(pos_item_score,pos_item_pop_prod/user_pop_n2/pos_item_pop_n2/2+0.5)/self.tau
        #weighted_neg_item_score=tf.multiply(neg_item_score,neg_item_pop_prod/tiled_usr_pop_n2/neg_item_pop_n2/2+0.5)/self.tau
        neg_item_score_exp=tf.reduce_sum(tf.exp(tf.reshape(weighted_neg_item_score,[-1,self.neg_sample])),axis=1)
        pos_item_score_exp=tf.exp(weighted_pos_item_score)
        loss1=(1-self.w_lambda)*tf.reduce_mean(tf.negative(tf.log(pos_item_score_exp/(pos_item_score_exp+neg_item_score_exp))))

        regularizer1 = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer1 = regularizer1/self.batch_size

        regularizer2=  tf.nn.l2_loss(users_pop) + tf.nn.l2_loss(pos_items_pop) + tf.nn.l2_loss(neg_items_pop)
        regularizer2  = regularizer2/self.batch_size
        reg_loss = self.decay * (regularizer1+regularizer2)

        reg_loss_freeze=self.decay * (regularizer2)
        reg_loss_norm=self.decay * (regularizer1)

        return loss1, loss2, reg_loss, reg_loss_freeze, reg_loss_norm, loss_mf, debug
    
    def compute_loss_for_correlation(self,pos_score,neg_score,method="infonce"):
        if method=="infonce":
            neg_exp = np.sum(np.exp(np.reshape(neg_score,[-1,self.neg_sample])),axis=1)
            pos_exp=np.exp(pos_score)
            loss=-np.log(pos_exp/(pos_exp+neg_exp))
        else:
            loss=np.log(1/(1 + np.exp(neg_score-pos_score))+1e-10)
        return loss

    
    def get_score(self, users, pos_items, neg_items, method="infonce",loss="dyninfo"):
        if method=="infonce":
            if loss=="dyninfo":
                to_com=[self.ensemble_pos_score,self.ensemble_neg_score,self.pop_pos_score,self.pop_neg_score,self.pos_score,self.neg_score]
                data = self.sess.run(to_com, {self.users: users,
                                                            self.pos_items: pos_items,
                                                            self.neg_items: neg_items})
            else:
                to_com=[self.pos_score,self.neg_score]
                data = self.sess.run(to_com, {self.users: users,
                                                            self.pos_items: pos_items,
                                                            self.neg_items: neg_items})
        else:
            if loss=="dyninfo":
                to_com=[self.ensemble_pos_score_ori,self.ensemble_neg_score_ori,self.pop_pos_score_ori,self.pop_neg_score_ori,self.pos_score_ori,self.neg_score_ori]
                data = self.sess.run(to_com, {self.users: users,
                                                            self.pos_items: pos_items,
                                                            self.neg_items: neg_items})
            else:
                to_com=[self.pos_score_ori,self.neg_score_ori]
                data = self.sess.run(to_com, {self.users: users,
                                                            self.pos_items: pos_items,
                                                            self.neg_items: neg_items})


        return data
            
        




    def create_bce_loss_two_brach1(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)   #users, pos_items, neg_items have the same shape
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        # item score
        # pos_items_stop = tf.stop_gradient(pos_items)
        # neg_items_stop = tf.stop_gradient(neg_items)
        pos_items_stop = pos_items
        neg_items_stop = neg_items
        self.pos_item_scores = tf.matmul(pos_items_stop, self.w)
        self.neg_item_scores = tf.matmul(neg_items_stop, self.w)
        self.rubi_ratings1 = (self.batch_ratings-self.rubi_c)*tf.squeeze(tf.nn.sigmoid(self.pos_item_scores))
        self.direct_minus_ratings1 = self.batch_ratings-self.rubi_c*tf.squeeze(tf.nn.sigmoid(self.pos_item_scores))
        # first branch
        # fusion
        pos_scores = pos_scores*tf.nn.sigmoid(self.pos_item_scores)
        neg_scores = neg_scores*tf.nn.sigmoid(self.neg_item_scores)
        self.mf_loss_ori = tf.reduce_mean(tf.negative(tf.log(tf.nn.sigmoid(pos_scores)+1e-10))+tf.negative(tf.log(1-tf.nn.sigmoid(neg_scores)+1e-10)))
        # second branch
        self.mf_loss_item = tf.reduce_mean(tf.negative(tf.log(tf.nn.sigmoid(self.pos_item_scores)+1e-10))+tf.negative(tf.log(1-tf.nn.sigmoid(self.neg_item_scores)+1e-10)))
        # unify
        mf_loss = self.mf_loss_ori + self.alpha*self.mf_loss_item
        # regular
        regularizer = tf.nn.l2_loss(self.u_g_embeddings_pre) + tf.nn.l2_loss(
                self.pos_i_g_embeddings_pre) + tf.nn.l2_loss(self.neg_i_g_embeddings_pre)
        regularizer = regularizer/self.batch_size
        emb_loss = self.decay * regularizer

        reg_loss = tf.constant(0.0, tf.float32, [1])

        return mf_loss, emb_loss, reg_loss

    def create_bce_loss_two_brach2(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)   #users, pos_items, neg_items have the same shape
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        # item score
        # pos_items_stop = tf.stop_gradient(self.pos_i_g_embeddings_pre)
        # neg_items_stop = tf.stop_gradient(self.neg_i_g_embeddings_pre)
        pos_items_stop = self.pos_i_g_embeddings_pre
        neg_items_stop = self.neg_i_g_embeddings_pre
        self.pos_item_scores = tf.matmul(pos_items_stop, self.w)
        self.neg_item_scores = tf.matmul(neg_items_stop, self.w)
        self.rubi_ratings2 = (self.batch_ratings-self.rubi_c)*tf.squeeze(tf.nn.sigmoid(self.pos_item_scores))
        self.direct_minus_ratings2 = self.batch_ratings-self.rubi_c*tf.squeeze(tf.nn.sigmoid(self.pos_item_scores))
        # first branch
        # fusion
        pos_scores = pos_scores*tf.nn.sigmoid(self.pos_item_scores)
        neg_scores = neg_scores*tf.nn.sigmoid(self.neg_item_scores)
        self.mf_loss_ori = tf.reduce_mean(tf.negative(tf.log(tf.nn.sigmoid(pos_scores)+1e-10))+tf.negative(tf.log(1-tf.nn.sigmoid(neg_scores)+1e-10)))
        # second branch
        self.mf_loss_item = tf.reduce_mean(tf.negative(tf.log(tf.nn.sigmoid(self.pos_item_scores)+1e-10))+tf.negative(tf.log(1-tf.nn.sigmoid(self.neg_item_scores)+1e-10)))
        # unify
        mf_loss = self.mf_loss_ori + self.alpha*self.mf_loss_item
        # regular
        regularizer = tf.nn.l2_loss(self.u_g_embeddings_pre) + tf.nn.l2_loss(
                self.pos_i_g_embeddings_pre) + tf.nn.l2_loss(self.neg_i_g_embeddings_pre)
        regularizer = regularizer/self.batch_size
        emb_loss = self.decay * regularizer

        reg_loss = tf.constant(0.0, tf.float32, [1])

        return mf_loss, emb_loss, reg_loss


    def create_bce_loss_two_brach_both(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)   #users, pos_items, neg_items have the same shape
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        # item score
        # pos_items_stop = tf.stop_gradient(pos_items)
        # neg_items_stop = tf.stop_gradient(neg_items)
        pos_items_stop = pos_items
        neg_items_stop = neg_items
        users_stop = users
        self.pos_item_scores = tf.matmul(pos_items_stop, self.w)
        self.neg_item_scores = tf.matmul(neg_items_stop, self.w)
        self.user_scores = tf.matmul(users_stop, self.w_user)
        # self.rubi_ratings_both = (self.batch_ratings-self.rubi_c)*(tf.transpose(tf.nn.sigmoid(self.pos_item_scores))+tf.nn.sigmoid(self.user_scores))
        # self.direct_minus_ratings_both = self.batch_ratings-self.rubi_c*(tf.transpose(tf.nn.sigmoid(self.pos_item_scores))+tf.nn.sigmoid(self.user_scores))
        self.rubi_ratings_both = (self.batch_ratings-self.rubi_c)*tf.transpose(tf.nn.sigmoid(self.pos_item_scores))*tf.nn.sigmoid(self.user_scores)
        self.direct_minus_ratings_both = self.batch_ratings-self.rubi_c*tf.transpose(tf.nn.sigmoid(self.pos_item_scores))*tf.nn.sigmoid(self.user_scores)
        # first branch
        # fusion
        pos_scores = pos_scores*tf.nn.sigmoid(self.pos_item_scores)*tf.nn.sigmoid(self.user_scores)
        neg_scores = neg_scores*tf.nn.sigmoid(self.neg_item_scores)*tf.nn.sigmoid(self.user_scores)
        # pos_scores = pos_scores*(tf.nn.sigmoid(self.pos_item_scores)+tf.nn.sigmoid(self.user_scores))
        # neg_scores = neg_scores*(tf.nn.sigmoid(self.pos_item_scores)+tf.nn.sigmoid(self.user_scores))
        self.mf_loss_ori = tf.reduce_mean(tf.negative(tf.log(tf.nn.sigmoid(pos_scores)+1e-10))+tf.negative(tf.log(1-tf.nn.sigmoid(neg_scores)+1e-10)))
        # second branch
        self.mf_loss_item = tf.reduce_mean(tf.negative(tf.log(tf.nn.sigmoid(self.pos_item_scores)+1e-10))+tf.negative(tf.log(1-tf.nn.sigmoid(self.neg_item_scores)+1e-10)))
        # third branch
        self.mf_loss_user = tf.reduce_mean(tf.negative(tf.log(tf.nn.sigmoid(self.user_scores)+1e-10))+tf.negative(tf.log(1-tf.nn.sigmoid(self.user_scores)+1e-10)))
        # unify
        mf_loss = self.mf_loss_ori + self.alpha*self.mf_loss_item + self.beta*self.mf_loss_user
        # regular
        regularizer = tf.nn.l2_loss(self.u_g_embeddings_pre) + tf.nn.l2_loss(
                self.pos_i_g_embeddings_pre) + tf.nn.l2_loss(self.neg_i_g_embeddings_pre)
        regularizer = regularizer/self.batch_size
        emb_loss = self.decay * regularizer

        reg_loss = tf.constant(0.0, tf.float32, [1])

        return mf_loss, emb_loss, reg_loss



    
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)
        
    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        """
        Dropout for sparse tensors.
        """
        noise_shape = [n_nonzero_elems]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse_retain(X, dropout_mask)

        return pre_out * tf.div(1., keep_prob)
    
    def update_c(self, sess, c):
        sess.run(tf.assign(self.rubi_c, c*tf.ones([1])))


    def add_sess(self, sess, args):
        self.sess=sess
        self.method=args.test
    
    def set_method(self, method):
        self.method=method
    
    def predict(self, users, items=None):
        if items==None:
            items = list(range(self.n_items))
        if self.method=="normal":
            rate_batch = self.sess.run(self.batch_ratings, {self.users: users,
                                                        self.pos_items: items,
                                                        self.node_dropout: [0.] * len(self.weight_size),
                                                        self.mess_dropout: [0.] * len(self.weight_size)})
        elif self.method == 'causal':
            rate_batch = self.sess.run(self.batch_ratings_causal_c, {self.users: users,
                                                        self.pos_items: items,
                                                        self.node_dropout: [0.] * len(self.weight_size),
                                                        self.mess_dropout: [0.] * len(self.weight_size)})
        elif self.method == 'rubiboth':
            rate_batch = self.sess.run(self.rubi_ratings_both, {self.users: users,
                                                        self.pos_items: items,
                                                        self.node_dropout: [0.] * len(self.weight_size),
                                                        self.mess_dropout: [0.] * len(self.weight_size)})
        elif self.method == 'pop':
            rate_batch = self.sess.run(self.batch_ratings_pop, {self.users: users,
                                                        self.pos_items: items,
                                                        self.node_dropout: [0.] * len(self.weight_size),
                                                        self.mess_dropout: [0.] * len(self.weight_size)})
        elif self.method == 'most_pop':
            rate_batch = self.sess.run(self.pos_item_p, {self.users: users,
                                                        self.pos_items: items,
                                                        self.node_dropout: [0.] * len(self.weight_size),
                                                        self.mess_dropout: [0.] * len(self.weight_size)})
            rate_batch = np.tile(np.array(rate_batch),(len(users),1))


            

        return rate_batch


def load_pretrained_data():
    pretrain_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, 'embedding')
    try:
        pretrain_data = np.load(pretrain_path)
        print('load the pretrained embeddings.')
    except Exception:
        pretrain_data = None
    return pretrain_data

# parallelized sampling on CPU 
class sample_thread(threading.Thread):
    def __init__(self,pop=0,inbatch=1):
        self.pop=pop
        self.inbatch=inbatch
        threading.Thread.__init__(self)
    def run(self):
        if self.pop==0:
            with tf.device(cpus[0]):
                self.data = data_generator.sample()
        elif self.pop==1:
            if self.inbatch==1:
                with tf.device(cpus[0]):
                    self.data = data_generator.sample_infonce_inbatch(data_generator.user_pop_idx,data_generator.item_pop_idx)
            else:
                with tf.device(cpus[0]):
                    self.data = data_generator.sample_infonce(data_generator.user_pop_idx,data_generator.item_pop_idx)

        else:
            with tf.device(cpus[0]):
                self.data = data_generator.sample_cause()
        

class sample_thread_test(threading.Thread):
    def __init__(self,pop=0):
        self.pop=pop
        threading.Thread.__init__(self)
    def run(self):
        if not self.pop:
            with tf.device(cpus[0]):
                self.data = data_generator.sample_test()
        else:
            with tf.device(cpus[0]):
                self.data = data_generator.sample_infonce(data_generator.user_pop_idx,data_generator.item_pop_idx)
            
# training on GPU
class train_thread(threading.Thread):
    def __init__(self,model, sess, sample, args, epoch):
        threading.Thread.__init__(self)
        self.epoch=epoch
        self.model = model
        self.sess = sess
        self.sample = sample
        self.args=args
    def run(self):
        sess_list = []
        if args.loss == 'bpr':
            sess_list = [self.model.opt, self.model.loss, self.model.mf_loss, self.model.emb_loss, self.model.reg_loss]
        elif args.loss == 'bce':
            sess_list = [self.model.opt_bce, self.model.loss_bce, self.model.mf_loss_bce, self.model.emb_loss_bce, self.model.reg_loss_bce]
        elif args.loss == 'bce1':
            sess_list = [self.model.opt_two_bce1, self.model.loss_two_bce1, self.model.mf_loss_two_bce1, self.model.emb_loss_two_bce1, self.model.reg_loss_two_bce1]
        elif args.loss == 'bce2':
            sess_list = [self.model.opt_two_bce2, self.model.loss_two_bce2, self.model.mf_loss_two_bce2, self.model.emb_loss_two_bce2, self.model.reg_loss_two_bce2]
        elif args.loss == 'bceboth':
            sess_list = [self.model.opt_two_bce_both, self.model.loss_two_bce_both, self.model.mf_loss_two_bce_both, self.model.emb_loss_two_bce_both, self.model.reg_loss_two_bce_both]
        elif args.loss == 'ipw':
            sess_list = [self.model.opt_ipw, self.model.ipw_loss, self.model.ipw_mf_loss, self.model.ipw_reg_loss, self.model.reg_loss]
        elif args.loss == 'dyninfo':
            if self.args.freeze==1:
                if self.epoch <= self.args.freeze_epoch:
                    sess_list = [self.model.opt_freeze, self.model.loss_info, self.model.mf_loss_info_1, self.model.mf_loss_info_2, self.model.reg_loss_info]#,self.model.debug]
                else:
                    sess_list = [self.model.opt_none_freeze, self.model.loss_info, self.model.mf_loss_info_1, self.model.mf_loss_info_2, self.model.reg_loss_info]#,self.model.debug]
            else:
                sess_list = [self.model.opt_info, self.model.loss_info, self.model.mf_loss_info_1, self.model.mf_loss_info_2, self.model.reg_loss_info]
        elif args.loss == 'mf_info':
            sess_list = [self.model.opt_mf_info, self.model.loss_info, self.model.mf_loss_info_1, self.model.mf_loss_info_2, self.model.reg_loss_info]
        elif args.loss == 'CausE':
            sess_list = [self.model.opt_cause, self.model.loss_cause, self.model.cause_mf_loss, self.model.cause_cf_loss, self.model.cause_reg_loss]

        if args.loss=="dyninfo" or args.loss=="mf_info":
            if len(gpus):
                with tf.device(gpus[-1]):
                    users, pos_items, neg_items, users_pop, pos_items_pop, neg_items_pop = self.sample.data
                    self.data = sess.run(sess_list,
                                        feed_dict={model.users: users, model.pos_items: pos_items,
                                                    model.node_dropout: eval(args.node_dropout),
                                                    model.mess_dropout: eval(args.mess_dropout),
                                                    model.neg_items: neg_items,
                                                    model.users_pop: users_pop,
                                                    model.pos_items_pop: pos_items_pop,
                                                    model.neg_items_pop: neg_items_pop
                                                    })
            else:
                users, pos_items, neg_items, users_pop, pos_items_pop, neg_items_pop = self.sample.data
                self.data = sess.run(sess_list,
                                        feed_dict={model.users: users, model.pos_items: pos_items,
                                                    model.node_dropout: eval(args.node_dropout),
                                                    model.mess_dropout: eval(args.mess_dropout),
                                                    model.neg_items: neg_items,
                                                    model.users_pop: users_pop,
                                                    model.pos_items_pop: pos_items_pop,
                                                    model.neg_items_pop: neg_items_pop})
        elif args.loss=="CausE":
            if len(gpus):
                with tf.device(gpus[-1]):
                    users, pos_items, neg_items, all_items, ctrl_items = self.sample.data
                    self.data = sess.run(sess_list,
                                        feed_dict={model.users: users, model.pos_items: pos_items,
                                                    model.node_dropout: eval(args.node_dropout),
                                                    model.mess_dropout: eval(args.mess_dropout),
                                                    model.neg_items: neg_items,
                                                    model.items: all_items,
                                                    model.ctrl_items: ctrl_items})
            else:
                users, pos_items, neg_items, all_items, ctrl_items = self.sample.data
                self.data = sess.run(sess_list,
                                        feed_dict={model.users: users, model.pos_items: pos_items,
                                                    model.node_dropout: eval(args.node_dropout),
                                                    model.mess_dropout: eval(args.mess_dropout),
                                                    model.neg_items: neg_items,
                                                    model.items: all_items,
                                                    model.ctrl_items: ctrl_items})

            
        else:
            if len(gpus):
                with tf.device(gpus[-1]):
                    users, pos_items, neg_items = self.sample.data
                    self.data = sess.run(sess_list,
                                        feed_dict={model.users: users, model.pos_items: pos_items,
                                                    model.node_dropout: eval(args.node_dropout),
                                                    model.mess_dropout: eval(args.mess_dropout),
                                                    model.neg_items: neg_items})
            else:
                users, pos_items, neg_items = self.sample.data
                self.data = sess.run(sess_list,
                                        feed_dict={model.users: users, model.pos_items: pos_items,
                                                    model.node_dropout: eval(args.node_dropout),
                                                    model.mess_dropout: eval(args.mess_dropout),
                                                    model.neg_items: neg_items})
            
class train_thread_test(threading.Thread):
    def __init__(self, model, sess, sample, args):
        threading.Thread.__init__(self)
        self.model = model
        self.sess = sess
        self.sample = sample
    def run(self):
        sess_list = []
        if args.loss == 'bpr':
            sess_list = [self.model.loss, self.model.mf_loss, self.model.emb_loss, self.model.reg_loss]
        elif args.loss == 'bce':
            sess_list = [self.model.loss_bce, self.model.mf_loss_bce, self.model.emb_loss_bce, self.model.reg_loss_bce]
        elif args.loss == 'bce1':
            sess_list = [self.model.loss_two_bce1, self.model.mf_loss_two_bce1, self.model.emb_loss_two_bce1, self.model.reg_loss_two_bce1]
        elif args.loss == 'bce2':
            sess_list = [self.model.loss_two_bce2, self.model.mf_loss_two_bce2, self.model.emb_loss_two_bce2, self.model.reg_loss_two_bce2]
        elif args.loss == 'bceboth':
            sess_list = [self.model.loss_two_bce_both, self.model.mf_loss_two_bce_both, self.model.emb_loss_two_bce_both, self.model.reg_loss_two_bce_both]
        elif args.loss == 'dyninfo':
            sess_list = [self.model.loss_info, self.model.mf_loss_info_1, self.model.mf_loss_info_2, self.model.reg_loss_info]

        if args.loss!="dyninfo":
            if len(gpus):
                with tf.device(gpus[-1]):
                    users, pos_items, neg_items = self.sample.data
                    self.data = sess.run(sess_list,
                                        feed_dict={model.users: users, model.pos_items: pos_items,
                                                    model.neg_items: neg_items,
                                                    model.node_dropout: eval(args.node_dropout),
                                                    model.mess_dropout: eval(args.mess_dropout)})
            else:
                users, pos_items, neg_items = self.sample.data
                self.data = sess.run(sess_list,
                                        feed_dict={model.users: users, model.pos_items: pos_items,
                                                    model.neg_items: neg_items,
                                                    model.node_dropout: eval(args.node_dropout),
                                                    model.mess_dropout: eval(args.mess_dropout)})
        else:
            if len(gpus):
                with tf.device(gpus[-1]):
                    users, pos_items, neg_items, users_pop, pos_items_pop, neg_items_pop = self.sample.data
                    self.data = sess.run(sess_list,
                                        feed_dict={model.users: users, model.pos_items: pos_items,
                                                    model.node_dropout: eval(args.node_dropout),
                                                    model.mess_dropout: eval(args.mess_dropout),
                                                    model.neg_items: neg_items,
                                                    model.users_pop: users_pop,
                                                    model.pos_items_pop: pos_items_pop,
                                                    model.neg_items_pop: neg_items_pop
                                                    })
            else:
                users, pos_items, neg_items, users_pop, pos_items_pop, neg_items_pop = self.sample.data
                self.data = sess.run(sess_list,
                                        feed_dict={model.users: users, model.pos_items: pos_items,
                                                    model.node_dropout: eval(args.node_dropout),
                                                    model.mess_dropout: eval(args.mess_dropout),
                                                    model.neg_items: neg_items,
                                                    model.users_pop: users_pop,
                                                    model.pos_items_pop: pos_items_pop,
                                                    model.neg_items_pop: neg_items_pop})

def merge_user_list(user_lists):
    out=collections.defaultdict(list)
    for user_list in user_lists:
        for key, item in user_list.items():
            out[key]=out[key]+item
    return out

def norm01(a):
    return (a - np.min(a))/np.ptp(a)

if __name__ == '__main__':
    data=data_generator
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    

    f0 = time()
    
    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    Ks=eval(args.Ks)

    

    if "new" in args.dataset:
        eval_test_ood = ProxyEvaluator(data,data.train_user_list,data.test_user_list,top_k=Ks,dump_dict=merge_user_list([data.train_user_list,data.valid_user_list,data.test_id_user_list]))
        eval_test_id = ProxyEvaluator(data,data.train_user_list,data.test_id_user_list,top_k=Ks,dump_dict=merge_user_list([data.train_user_list,data.valid_user_list,data.test_user_list]))
        eval_valid = ProxyEvaluator(data,data.train_user_list,data.valid_user_list,top_k=Ks)
    else:
        eval_test_ood = ProxyEvaluator(data,data.train_user_list,data.test_user_list,top_k=Ks)
        eval_test_id = None
        eval_valid = eval_test_ood

    
    p_matrix = dict()
    p = []
    pop = []
    for item, users in data.train_item_list.items():
        p_matrix[item] = len(users)+1
    for item in data.items:
        # print(item)
        if item not in p_matrix.keys():
            p_matrix[item] = 1
        p.append(p_matrix[item]/(data.n_users+1))
        pop.append(p_matrix[item]-1)

    
    p_user = np.array([len(data.train_user_list[u]) if u in data.train_user_list else 0 for u in range(data.n_users)])
    # normal
    
    overlap_num=[10,20,30,40,50]
    poptest=[{},{},{},{},{}]
    p_all=np.array(pop)
    _mean=np.mean(p_all)
    _median=np.median(p_all)
    _max=np.max(p_all)

    print("distinct users pop:",len(np.unique(p_user)))
    print("distinct items pop:",len(np.unique(p_all))-1)
    print("distinct users:",len(p_user))
    print("distinct items:",len(p_all))
    print(_mean)
    print(_median)
    print(_max)
    #exit()

    #amazon:450 415

    #yelp 327 396

    #tencent:

    #ifashion:

    """
    *********************************************************
    Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
    """
    plain_adj, norm_adj, mean_adj, pre_adj = data_generator.get_adj_mat()
    plain_adj_pop, norm_adj_pop, mean_adj_pop, pre_adj_pop= data_generator.get_adj_mat(is_pop=True)
    if args.adj_type == 'plain':
        config['norm_adj'] = plain_adj
        config['norm_adj_pop']=plain_adj_pop
        print('use the plain adjacency matrix')
    elif args.adj_type == 'norm':
        config['norm_adj'] = norm_adj
        config['norm_adj_pop']=norm_adj_pop
        print('use the normalized adjacency matrix')
    elif args.adj_type == 'gcmc':
        config['norm_adj'] = mean_adj
        config['norm_adj_pop']=mean_adj_pop
        print('use the gcmc adjacency matrix')
    elif args.adj_type =='pre':
        config['norm_adj']=pre_adj
        config['norm_adj_pop']=pre_adj_pop
        print('use the pre adjcency matrix')
    else:
        config['norm_adj'] = mean_adj + sp.eye(mean_adj.shape[0])
        print('use the mean adjacency matrix')
    t0 = time()
    if args.pretrain == -1:
        pretrain_data = load_pretrained_data()
    else:
        pretrain_data = None
    model = LightGCN(data_config=config, pretrain_data=pretrain_data,user_pop_num=data_generator.user_pop_num, item_pop_num=data_generator.item_pop_num,user_pop=data_generator.user_pop_idx, item_pop=data_generator.item_pop_idx, pop_branch=args.pop_branch, pop_reduct=args.pop_reduct,p_matrix=p)
    model.add_sess(sess,args)
    
    """
    *********************************************************
    Save the model parameters.
    """
    saver = tf.train.Saver()

    #if args.save_flag == 1:
    layer = '-'.join([str(l) for l in eval(args.layer_size)])
    weights_save_path = '%sweights/%s/%s/%s/l%s_r%s/' % (args.weights_path, args.dataset, model.model_type, layer,
                                                        str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))
    ensureDir(weights_save_path)
    save_saver = tf.train.Saver(max_to_keep=5)

    """
    *********************************************************
    Reload the pretrained model parameters.
    """
    
    if args.pretrain == 1:

        p_user = np.array([len(data.train_user_list[u]) if u in data.train_user_list else 0 for u in range(data.n_users)])
        # normal
        overlap_num=[10,20,30,40,50]
        poptest=[{},{},{},{},{}]
        p_all=np.array(pop)
        _mean=np.mean(p_all)
        _median=np.median(p_all)
        _max=np.max(p_all)
        print(_mean)
        print(_median)
        print(_max)
        pop_sorted=np.sort(p_user)
        n_groups=2
        grp_view=[]
        for grp in range(n_groups):
            split=int((data.n_users-1)*(grp+1)/n_groups)
            grp_view.append(pop_sorted[split])
        print("group_view:",grp_view)
        #12,12 [mean,median]
        #42,15
        #19,21
        #12,3


        # rank=np.argsort(-p_all)
        # for user in range(data.n_users):
        #     dump=[]
        #     if user in data.train_user_list:
        #         dump+=data.train_user_list[user]
        #     if user in data.valid_user_list:
        #         dump+=data.valid_user_list[user]

        #     ptr=0
        #     num=0
        #     my_rank=[]
        #     while num<50:
        #         if rank[ptr] not in dump:
        #             my_rank.append(rank[ptr])
        #             num+=1
        #         ptr+=1


            
        #     for i,num in enumerate(overlap_num):
        #         poptest[i][user]=my_rank[:num]
        

        
                
        
        

        # model_file = "../weights/addressa/LightGCN/64-64/l0.001_r1e-05-1e-05-0.01/weights_gcnnormal-300"
        # users_to_test = list(data_generator.test_set.keys())
        # saver.restore(sess, model_file)
        # ret = test(sess, model, users_to_test)
        # perf_str = 'recall={}, hit={}, ndcg={}'.format(str(ret["recall"]),
        #                                     str(ret['hr']), str(ret['ndcg']))
        # print(perf_str)
        # exit()

        # MACR
       
        #model_file = "/storage/wcma/MACR/weights/yelp2018.new/LightGCN/l0.001_r1e-05-1e-05-0.01/weights_mf_popgo_not_inbatch_neg64_final_70"
        #model_file = "/storage/wcma/MACR/weights/yelp2018.new/LightGCN/l0.001_r1e-05-1e-05-0.01/weights_mf_infonce_final_rerun_466"
        #model_file = "/storage/wcma/MACR/weights/yelp2018.new/LightGCN/64-64/l0.001_r1e-05-1e-05-0.01/weights_mf_popgo_inbatch_final_25"
        #model_file="/storage/wcma/MACR/weights/yelp2018.new/LightGCN/64-64/l0.001_r1e-05-1e-05-0.01/weights_mf_bpr_final_112"
        
        #model_file = "/storage/wcma/MACR/weights/tencent.new/LightGCN/l0.001_r1e-05-1e-05-0.01/weights_mf_popgo_not_inbatch_neg64_final_108"
        #model_file="/storage/wcma/MACR/weights/tencent.new/LightGCN/l0.001_r1e-05-1e-05-0.01/weights_mf_bpr_final_24"
        #model_file="/storage/wcma/MACR/weights/tencent.new/LightGCN/l0.001_r1e-05-1e-05-0.01/weights_mf_infonce_final_rerun_278"
        #model_file="/storage/wcma/MACR/weights/tencent.new/LightGCN/64-64/l0.001_r1e-05-1e-05-0.01/weights_mf_popgo_inbatch_final_109"
        #model_file="/storage/wcma/MACR/weights/tencent.new/LightGCN/64-64/l0.001_r1e-05-1e-05-0.01/weights_mf_bpr_final_62"

        #model_file = "/storage/wcma/MACR/weights/amazon-book.new/LightGCN/l0.001_r1e-05-1e-05-0.01/weights_mf_popgo_not_inbatch_neg64_final_84"
        #model_file="/storage/wcma/MACR/weights/amazon-book.new/LightGCN/l0.001_r1e-05-1e-05-0.01/weights_mf_bpr_final_319"
        #model_file="/storage/wcma/MACR/weights/amazon-book.new/LightGCN/l0.001_r1e-05-1e-05-0.01/weights_mf_infonce_final_rerun_92"
        model_file="/storage/wcma/MACR/weights/amazon-book.new/LightGCN/64-64/l0.001_r1e-05-1e-05-0.01/weights_mf_bpr_final_180"
        #model_file="/storage/wcma/MACR/weights/amazon-book.new/LightGCN/64-64/l0.001_r1e-05-1e-05-0.01/weights_mf_popgo_inbatch_final_25"


        #model_file = "/storage/wcma/MACR/weights/ifashion.new/LightGCN/l0.001_r1e-05-1e-05-0.01/weights_mf_popgo_not_inbatch_neg64_final_204"
        #model_file = "/storage/wcma/MACR/weights/ifashion.new/LightGCN/l0.001_r1e-05-1e-05-0.01/weights_mf_infonce_final_rerun_276"
        #model_file="/storage/wcma/MACR/weights/ifashion.new/LightGCN/64-64/l0.001_r1e-05-1e-05-0.01/weights_mf_bpr_final_36"
        #model_file = "/storage/wcma/MACR/weights/ifashion.new/LightGCN/64-64/l0.001_r1e-05-1e-05-0.01/weights_mf_popgo_inbatch_final_25"
        # best_c = [30,50] # value of c
        # users_to_test = list(data_generator.test_set.keys())
        saver.restore(sess, model_file)

        # for i,num in enumerate(overlap_num):
        #     pop_eval=ProxyEvaluator(data,data.train_user_list,poptest[i],top_k=[num],dump_dict=merge_user_list([data.train_user_list,data.valid_user_list]),group_view=grp_view)
        #     ret = pop_eval.evaluate(model)
        #     print("Overlap@",num,":")
        #     print(ret[:,1])

        # test_users = list(self.test_user_list.keys())
        # test_users = DataIterator(test_users, batch_size=args.batch_size, shuffle=False, drop_last=False)
        # batch_result = []

        method="infonce"
        print("Calculating correlation with loss: ",method)
        if args.loss!="dyninfo":
            users = np.load(data_generator.path + '/users_'+method+'.npy')
            pos_items = np.load(data_generator.path + '/pos_items_'+method+'.npy')
            neg_items = np.load(data_generator.path + '/neg_items_'+method+'.npy')
        else:
            print("shit!")
            users, pos_items, neg_items=data_generator.sample_infonce_test(data_generator.user_pop_idx,data_generator.item_pop_idx,method=method)
            users=np.array(users)
            pos_items=np.array(pos_items)
            neg_items=np.array(neg_items)
            np.save(data_generator.path + '/users_'+method+'.npy', users)
            np.save(data_generator.path + '/pos_items_'+method+'.npy', pos_items)
            np.save(data_generator.path + '/neg_items_'+method+'.npy', neg_items)

        
        data=model.get_score(users,pos_items,neg_items,loss=args.loss,method=method)
        for i in range(len(data)):
            data[i]=np.array(data[i])
        
        if args.loss=="dyninfo":

            ensemble_pos_score,ensemble_neg_score,pop_pos_score,pop_neg_score,pos_score,neg_score=data
            s_e=model.compute_loss_for_correlation(ensemble_pos_score,ensemble_neg_score,method)
            s_o=model.compute_loss_for_correlation(pos_score,neg_score,method)
            s_p=model.compute_loss_for_correlation(pop_pos_score,pop_neg_score,method)

            

            print("ensemble vs. pop(score):",np.corrcoef(norm01(ensemble_pos_score),norm01(pop_pos_score))[0,1])
            print("debiased vs. pop(score):",np.corrcoef(norm01(pos_score),norm01(pop_pos_score))[0,1])
            print("ensemble vs. pop:",np.corrcoef(s_e,s_p)[0,1])
            print("debiased vs. pop:",np.corrcoef(s_o,s_p)[0,1])

            np.save(data_generator.path + '/pop_pos_score_'+method+'.npy',pop_pos_score)
            np.save(data_generator.path + '/pop_neg_score_'+method+'.npy',pop_neg_score)

        else:
            pop_pos_score=np.load(data_generator.path + '/pop_pos_score_'+method+'.npy')
            pop_neg_score=np.load(data_generator.path + '/pop_neg_score_'+method+'.npy')
            pos_score,neg_score=data

            s_m=model.compute_loss_for_correlation(pos_score,neg_score,method)
            s_p=model.compute_loss_for_correlation(pop_pos_score,pop_neg_score,method)

            print("mf vs. pop(score):",np.corrcoef(norm01(pos_score),norm01(pop_pos_score))[0,1])
            print("mf vs. pop:",np.corrcoef(s_m,s_p)[0,1])

        
        #print(self.pop_mask)


        # items = list(range(model.n_items))
        # item_embed,graph_embed,pop_embed=sess.run([model.weights['item_embedding'],model.pos_i_g_embeddings,model.pos_item_pop_embedding],
        #                                                                                 {model.pos_items: items,
        #                                                                                 model.node_dropout: [0.] * len(model.weight_size),
        #                                                                                 model.mess_dropout: [0.] * len(model.weight_size)})
        # pop=np.array(pop)
        # print(np.array(item_embed).shape)
        # np.save('tencent_infonce_embed.npy',np.array(item_embed))
        
        #np.save('tencent_pre_embed_info_lightgcn.npy',np.array(item_embed))
        #np.save('tencent_graph_embed_info_lightgcn.npy',np.array(graph_embed))
        #np.save('tencent_pop_embed_info_lightgcn.npy',np.array(pop_embed))
        # np.save('tencent_pop.npy',pop)
        #plot_tsne_embed(item_embed,pop)
        # for c in best_c:
        #     model.update_c(sess, c)
        #     names=["valid","test_ood","test_id"]
        #     test_trials=[eval_valid,eval_test_ood,eval_test_id]
        #     for w,_eval in enumerate(test_trials):
        #         ret, _ = _eval.evaluate(model)
        #         t3 = time()
        #         n_ret={"recall":[ret[1],ret[1]],"hit_ratio":[ret[5],ret[5]],"precision":[ret[0],ret[0]],"ndcg":[ret[4],ret[4]]}
        #         #["Precision", "Recall", "MAP", "NDCG", "MRR", "HR"]
        #         ret=n_ret

        #         if args.verbose > 0:
        #             perf_str = 'c:%.2f recall=[%.5f, %.5f], ' \
        #                         'hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]\n' % \
        #                         (c, ret['recall'][0], ret['recall'][-1],
        #                             ret['hit_ratio'][0], ret['hit_ratio'][-1],
        #                             ret['ndcg'][0], ret['ndcg'][-1])
        #             print(perf_str)
        #             with open(weights_save_path + 'stats_{}.txt'.format(args.saveID),'a') as f:
        #                 f.write(perf_str+"\n")
        #             print(perf_str)
        exit()

    elif args.pretrain == 0:
        sess.run(tf.global_variables_initializer())
        cur_best_pre_0 = 0.
        print('without pretraining.')

    
    if args.loss=="most_pop":
        model.set_method('most_pop')
        names=["valid","test_ood","test_id"]
        test_trials=[eval_valid,eval_test_ood,eval_test_id]
        for w,_eval in enumerate(test_trials):
            ret, _ = _eval.evaluate(model)
            t3 = time()
            n_ret={"recall":[ret[1],ret[1]],"hit_ratio":[ret[5],ret[5]],"precision":[ret[0],ret[0]],"ndcg":[ret[4],ret[4]]}
            #["Precision", "Recall", "MAP", "NDCG", "MRR", "HR"]
            ret=n_ret

            if args.verbose > 0:
                perf_str = '%s: recall=[%.5f, %.5f], ' \
                            'hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]\n' % \
                            (names[w], ret['recall'][0], ret['recall'][-1],
                                ret['hit_ratio'][0], ret['hit_ratio'][-1],
                                ret['ndcg'][0], ret['ndcg'][-1])
                print(perf_str)
        exit()


    """
    *********************************************************
    Train.
    """
    tensorboard_model_path = 'tensorboard/'
    if not os.path.exists(tensorboard_model_path):
        os.makedirs(tensorboard_model_path)
    run_time = 1
    while (True):
        if os.path.exists(tensorboard_model_path + model.log_dir +'/run_' + str(run_time)):
            run_time += 1
        else:
            break
    train_writer = tf.summary.FileWriter(tensorboard_model_path +model.log_dir+ '/run_' + str(run_time), sess.graph)
    
    
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    config = dict()
    config["best_hr"], config["best_ndcg"], config['best_recall'], config['best_pre'], config["best_epoch"] = 0, 0, 0, 0, 0
    config['best_c_hr'], config['best_c_epoch'] = 0, 0
    stopping_step = 0
    should_stop = False
    
    best_epoch=0
    best_hr_norm = 0
    best_str = ''
    
    if args.loss=="dyninfo" or args.loss=="mf_info":
        is_pop=1
    elif args.loss=="CausE":
        is_pop=2
    else:
        is_pop=0

    if args.loss!="dyninfo":
        freeze = 0
    else:
        freeze = args.freeze_epoch
        #model.set_method("pop")
    data_generator.check()
    if args.only_test == 0 and args.pretrain == 0:
        for epoch in range(1, args.epoch +1):
            t1 = time()
            loss, mf_loss, emb_loss, reg_loss = 0., 0., 0., 0.
            n_batch = data_generator.n_train // args.batch_size + 1
            loss_test,mf_loss_test,emb_loss_test,reg_loss_test=0.,0.,0.,0.
            '''
            *********************************************************
            parallelized train sampling
            '''
            sample_last = sample_thread(pop=is_pop,inbatch=args.inbatch_sample)
            sample_last.start()
            sample_last.join()
            for idx in tqdm(range(n_batch)):
                train_cur = train_thread(model, sess, sample_last, args, epoch)
                sample_next = sample_thread(pop=is_pop,inbatch=args.inbatch_sample)
                
                train_cur.start()
                sample_next.start()
                
                sample_next.join()
                train_cur.join()
                #users, pos_items, neg_items, users_pop = sample_last.data
                _, batch_loss, batch_mf_loss, batch_emb_loss, batch_reg_loss= train_cur.data
                sample_last = sample_next
            
                loss += batch_loss/n_batch
                mf_loss += batch_mf_loss/n_batch
                emb_loss += batch_emb_loss/n_batch
                reg_loss += batch_reg_loss/n_batch
            

            # print(debug)
            # print(np.mean(debug))
            # print(np.var(debug))
            #input()              
            if np.isnan(loss) == True:
                if args.verbose > 0 and epoch % args.verbose == 0:
                    perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f]' % (
                        epoch, time() - t1, loss, mf_loss, emb_loss, reg_loss)
                    print(perf_str)
                print('ERROR: loss is nan.')
                sys.exit()
            if (epoch % args.log_interval) != 0 or epoch <= freeze :
                if args.verbose > 0 and epoch % args.verbose == 0:
                    perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f]' % (
                        epoch, time() - t1, loss, mf_loss, emb_loss, reg_loss)
                    with open(weights_save_path + 'stats_{}.txt'.format(args.saveID),'a') as f:
                        f.write(perf_str+"\n")
                    print(perf_str)
                continue
            # if epoch > args.freeze_epoch:
            #     model.set_method(args.test)
            # users_to_test = list(data_generator.train_items.keys())
            # ret = test(sess, model, users_to_test ,drop_flag=True,train_set_flag=1)
            # perf_str = 'Epoch %d: train==[%.5f=%.5f + %.5f + %.5f], recall=[%s], hr=[%s], ndcg=[%s]' % \
            #            (epoch, loss, mf_loss, emb_loss, reg_loss, 
            #             ', '.join(['%.5f' % r for r in ret['recall']]),
            #             ', '.join(['%.5f' % r for r in ret['hr']]),
            #             ', '.join(['%.5f' % r for r in ret['ndcg']]))
            # print(perf_str)
            # summary_train_acc = sess.run(model.merged_train_acc, feed_dict={model.train_rec_first: ret['recall'][0],
            #                                                                 model.train_rec_last: ret['recall'][-1],
            #                                                                 model.train_ndcg_first: ret['ndcg'][0],
            #                                                                 model.train_ndcg_last: ret['ndcg'][-1]})
            # train_writer.add_summary(summary_train_acc, epoch // 20)


            t2 = time()
            #users_to_test = list(data.test_user_list.keys())
            if args.test=="rubiboth":
                best_hr=0
                best_c=0
                for c in np.linspace(args.start, args.end, args.step):
                    model.update_c(sess, c)
                    if "new" in args.dataset: 
                        names=["valid","test_ood","test_id"]
                        test_trials=[eval_valid,eval_test_ood,eval_test_id]
                    else:
                        names=["valid"]
                        test_trials=[eval_valid]

                    for w,_eval in enumerate(test_trials):
                        ret, _ = _eval.evaluate(model)
                        t3 = time()

                        n_ret={"recall":[ret[1],ret[1]],"hit_ratio":[ret[5],ret[5]],"precision":[ret[0],ret[0]],"ndcg":[ret[4],ret[4]]}
                        #["Precision", "Recall", "MAP", "NDCG", "MRR", "HR"]
                        ret=n_ret
                        if w==0:
                            rec_loger.append(ret['recall'][0])
                            ndcg_loger.append(ret['ndcg'][0])
                            hit_loger.append(ret['hit_ratio'][0])

                            if ret['hit_ratio'][0] > best_hr:
                                best_hr = ret['hit_ratio'][0]
                                best_c = c

                        if args.verbose > 0:
                            perf_str = 'c:%.2f recall=[%.5f, %.5f], ' \
                                        'hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]\n' % \
                                        (c, ret['recall'][0], ret['recall'][-1],
                                            ret['hit_ratio'][0], ret['hit_ratio'][-1],
                                            ret['ndcg'][0], ret['ndcg'][-1])
                            print(perf_str)
                            with open(weights_save_path + 'stats_{}.txt'.format(args.saveID),'a') as f:
                                f.write(perf_str+"\n")
                model.update_c(sess, best_c)
            
            else:
                if "new" in args.dataset: 
                    names=["valid","test_ood","test_id"]
                    test_trials=[eval_valid,eval_test_ood,eval_test_id]
                else:
                    names=["valid"]
                    test_trials=[eval_valid]
                

                for w,_eval in enumerate(test_trials):
                    ret, _ = _eval.evaluate(model)
                    t3 = time()

                    n_ret={"recall":[ret[1],ret[1]],"hit_ratio":[ret[5],ret[5]],"precision":[ret[0],ret[0]],"ndcg":[ret[4],ret[4]]}
                    #["Precision", "Recall", "MAP", "NDCG", "MRR", "HR"]
                    ret=n_ret
                    if w==0:
                        rec_loger.append(ret['recall'][0])
                        pre_loger.append(ret['precision'][0])
                        ndcg_loger.append(ret['ndcg'][0])
                        hit_loger.append(ret['hit_ratio'][0])

                        
                        best_hr = ret['hit_ratio'][0]
                        best_recall=ret['recall'][0]
                        best_pre=ret['precision'][0]
                        best_ndcg=ret['ndcg'][0]

                    if args.verbose > 0:
                        perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f + %.5f], split=[%s], recall=[%.5f, %.5f], ' \
                            'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                            (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, reg_loss, names[w], ret['recall'][0], ret['recall'][-1],
                                ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                                ret['ndcg'][0], ret['ndcg'][-1])
                        print(perf_str)
                        with open(weights_save_path + 'stats_{}.txt'.format(args.saveID),'a') as f:
                            f.write(perf_str+"\n")
                
                
            # ret, _ = eval_valid.evaluate(model)
            # # ret = test(sess, model, users_to_test)
            # t3 = time()
            # print(ret)

            # n_ret={"recall":[ret[1],ret[1]],"hit_ratio":[ret[5],ret[5]],"precision":[ret[0],ret[0]],"ndcg":[ret[4],ret[4]]}
            # ret=n_ret
            # loss_loger.append(loss)
            # rec_loger.append(ret['recall'][0])
            # pre_loger.append(ret['precision'][0])
            # ndcg_loger.append(ret['ndcg'][0])
            # hit_loger.append(ret['hit_ratio'][0])
            # if args.verbose > 0:
            #     perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.8f=%.8f + %.8f], recall=[%.5f, %.5f], ' \
            #             'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
            #             (epoch, t2 - t1, t3 - t2, loss, mf_loss, reg_loss, ret['recall'][0], ret['recall'][-1],
            #                 ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
            #                 ret['ndcg'][0], ret['ndcg'][-1])
            #     print(perf_str)
            #     with open(weights_save_path + 'stats_{}.txt'.format(args.saveID),'a') as f:
            #         f.write(perf_str+"\n")
            
            # '''
            # *********************************************************
            # parallelized test sampling
            # '''
            # # test loss
            # sample_last= sample_thread_test(pop=is_pop)
            # sample_last.start()
            # sample_last.join()
            # loss_test,mf_loss_test,emb_loss_test,reg_loss_test=0.,0.,0.,0.
            # for idx in range(n_batch):
            #     train_cur = train_thread_test(model, sess, sample_last, args)
            #     sample_next = sample_thread_test(pop=is_pop)
                
            #     train_cur.start()
            #     sample_next.start()
                
            #     sample_next.join()
            #     train_cur.join()
                
            #     #users, pos_items, neg_items = sample_last.data
            #     batch_loss_test, batch_mf_loss_test, batch_emb_loss_test, batch_reg_loss_test = train_cur.data
            #     sample_last = sample_next
                
            #     loss_test += batch_loss_test / n_batch
            #     mf_loss_test += batch_mf_loss_test / n_batch
            #     emb_loss_test += batch_emb_loss_test / n_batch
            #     reg_loss_test += batch_reg_loss_test / n_batch
                
            # # summary_test_loss = sess.run(model.merged_test_loss,
            # #                             feed_dict={model.test_loss: loss_test, model.test_mf_loss: mf_loss_test,
            # #                                         model.test_emb_loss: emb_loss_test, model.test_reg_loss: reg_loss_test})
            # # train_writer.add_summary(summary_test_loss, epoch // 20)
            # t2 = time()
            # users_to_test = list(data_generator.test_set.keys())


            # perf_str = ''
            # if args.test == 'normal':
            #     ret = test(sess, model, users_to_test, drop_flag=True)                                                                                 
            #     t3 = time()
                
            #     loss_loger.append(loss)
            #     rec_loger.append(ret['recall'])
            #     pre_loger.append(ret['hr'])
            #     ndcg_loger.append(ret['ndcg'])

            #     if args.verbose > 0:
            #         perf_str = 'Epoch %d [%.1fs + %.1fs]: test==[%.5f=%.5f + %.5f + %.5f], recall=[%s], ' \
            #                 'hr=[%s], ndcg=[%s]\n' % \
            #                 (epoch, t2 - t1, t3 - t2, loss_test, mf_loss_test, emb_loss_test, reg_loss_test, 
            #                     ', '.join(['%.5f' % r for r in ret['recall']]),
            #                     ', '.join(['%.5f' % r for r in ret['hr']]),
            #                     ', '.join(['%.5f' % r for r in ret['ndcg']]))
            #         with open(weights_save_path + 'stats_{}.txt'.format(args.saveID),'a') as f:
            #             f.write(perf_str+"\n")
            #         print(perf_str, end='')
            #     if ret['hr'][0] > best_hr_norm:
            #         best_hr_norm = ret['hr'][0]
            #         best_epoch = epoch
            #         best_str = perf_str
            # elif args.test=="rubi1" or args.test=='rubi2' or args.test=='rubiboth':
            #     print('Epoch %d'%(epoch))
            #     best_c = 0
            #     best_hr = 0
            #     for c in np.linspace(args.start, args.end, args.step):
            #         model.update_c(sess, c)
            #         ret = test(sess, model, users_to_test, method=args.test)
            #         t3 = time()
            #         loss_loger.append(loss)
            #         rec_loger.append(ret['recall'][0])
            #         ndcg_loger.append(ret['ndcg'][0])
            #         hit_loger.append(ret['hr'][0])

            #         if ret['hr'][0] > best_hr:
            #             best_hr = ret['hr'][0]
            #             best_c = c

            #         if args.verbose > 0:
            #             perf_str += 'c:%.2f recall=[%.5f, %.5f], ' \
            #                         'hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]\n' % \
            #                         (c, ret['recall'][0], ret['recall'][-1],
            #                             ret['hr'][0], ret['hr'][-1],
            #                             ret['ndcg'][0], ret['ndcg'][-1])
                
            #     flg = False
            #     for c in np.linspace(best_c-1, best_c+1,6):
            #         model.update_c(sess, c)
            #         ret = test(sess, model, users_to_test, method=args.test)
            #         t3 = time()
            #         loss_loger.append(loss)
            #         rec_loger.append(ret['recall'][0])
            #         ndcg_loger.append(ret['ndcg'][0])
            #         hit_loger.append(ret['hr'][0])

            #         if ret['hr'][0] > best_hr:
            #             best_hr = ret['hr'][0]
            #             best_c = c

            #         if args.verbose > 0:
            #             perf_str += 'c:%.2f recall=[%.5f, %.5f], ' \
            #                         'hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]\n' % \
            #                         (c, ret['recall'][0], ret['recall'][-1],
            #                         ret['hr'][0], ret['hr'][-1],
            #                         ret['ndcg'][0], ret['ndcg'][-1])

            #         if ret['hr'][0] > config['best_c_hr']:
            #             config['best_c_hr'] = ret['hr'][0]
            #             config['best_c_ndcg'] = ret['ndcg'][0]
            #             config['best_c_recall'] = ret['recall'][0]
            #             config['best_c_epoch'] = epoch
            #             config['best_c'] = c
            #             flg = True
                    
            #     ret['hr'][0] = best_hr
            #     print(perf_str, end='')
            #     if flg:
            #         best_str = perf_str



            
                
            cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['ndcg'][0], cur_best_pre_0,
                                                                        stopping_step, expected_order='acc', flag_step=100)

            # *********************************************************
            # save the user & item embeddings for pretraining.
            if ret['ndcg'][0] == cur_best_pre_0:
                best_epoch = epoch
                if args.test!="normal":
                    config['best_c']=best_c
            if args.save_flag == 1:
                if best_epoch==epoch:
                    print('save the weights in path: ', weights_save_path)
                    save_saver.save(sess, weights_save_path + 'weights_{}_{}'.format(args.saveID, epoch))
                    config['best_name']=weights_save_path + 'weights_{}_{}'.format(args.saveID, epoch)
            # *********************************************************
            # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
            if should_stop == True and args.early_stop == 1:
                if args.test != 'normal':
                #     with open(weights_save_path + 'best_epoch_{}.txt'.format(args.saveID),'w') as f:
                #         f.write(str(config['best_c_epoch']))
                    with open(weights_save_path + 'best_c_{}.txt'.format(args.saveID),'w') as f:
                        f.write(str(config['best_c']))
                # else:
                with open(weights_save_path + 'best_epoch_{}.txt'.format(args.saveID),'w') as f:
                    f.write(str(best_epoch))
                break
        

        saver.restore(sess, config['best_name'])
        if args.test!='normal':
            model.update_c(sess, config['best_c'])
        
        ret, _ = eval_test_ood.evaluate(model)
        n_ret = {"recall":ret[1], "hit_ratio":ret[5], "precision":ret[0], "ndcg":ret[4]}
        ret=n_ret
        perf_str = 'test_ood: recall={}, ' \
                            'precision={}, hit={}, ndcg={}'.format(str(ret["recall"]),
                                str(ret['precision']), str(ret['hit_ratio']), str(ret['ndcg']))
        print(perf_str)
        with open(weights_save_path + 'stats_{}.txt'.format(args.saveID),'a') as f:
            f.write(perf_str+"\n")
        
        if "new" in args.dataset:
            ret, _ = eval_test_id.evaluate(model)
            n_ret = {"recall":ret[1], "hit_ratio":ret[5], "precision":ret[0], "ndcg":ret[4]}
            ret=n_ret
            perf_str = 'test_id: recall={}, ' \
                                'precision={}, hit={}, ndcg={}'.format(str(ret["recall"]),
                                    str(ret['precision']), str(ret['hit_ratio']), str(ret['ndcg']))
            print(perf_str)
            with open(weights_save_path + 'stats_{}.txt'.format(args.saveID),'a') as f:
                f.write(perf_str+"\n")

        # if args.test == 'rubi1' or args.test == 'rubi2' or args.test == 'rubiboth':
        #     print(config['best_c_epoch'], config['best_c_hr'], config['best_c_ndcg'], config['best_c_recall'],config['best_c'])
        # else:
        #     print(best_epoch, best_hr_norm)
        # print(best_str, end='')



    # if args.out == 1:
    #     best_epoch = 0
    #     best_c=0
    #     with open(weights_save_path+'/best_epoch_{}.txt'.format(args.saveID),'r') as f:
    #         best_epoch = eval(f.read())
    #     model_file = weights_save_path + '/weights_{}-{}'.format(args.saveID,best_epoch)
    #     save_saver.restore(sess, model_file)
    #     if args.test == 'rubiboth':
    #         with open(weights_save_path+'/best_c_{}.txt'.format(args.saveID),'r') as f:
    #             best_c = eval(f.read())
    #         model.update_c(sess, best_c)

    #         print(best_epoch, best_c)


    #     test_users = list(data_generator.test_set.keys())
    #     n_test_users = len(test_users)
    #     u_batch_size = BATCH_SIZE
    #     n_user_batchs = n_test_users // u_batch_size + 1
        
    #     total_rate = np.empty(shape=[0, ITEM_NUM])
    #     item_batch = list(range(ITEM_NUM))
    #     for u_batch_id in range(n_user_batchs):
    #         start = u_batch_id * u_batch_size
    #         end = (u_batch_id + 1) * u_batch_size

    #         user_batch = test_users[start: end]
    #         if args.test=="normal":
    #             rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
    #                                                             model.pos_items: item_batch})

    #         elif args.test == 'rubiboth':
    #             rate_batch = sess.run(model.rubi_ratings_both, {model.users: user_batch,
    #                                                                 model.pos_items: item_batch})

    #         total_rate = np.vstack((total_rate, rate_batch))
        
    #     total_sorted_id = np.argsort(-total_rate, axis=1)
    #     count = np.zeros(shape=[ITEM_NUM])
    #     for user, line in enumerate(total_sorted_id):
    #         # cutline = line[:10]
    #         # for item in cutline:
    #         #     count[item] += 1
    #         n = 0
    #         for item in line:
    #             if user not in data_generator.train_items.keys() or item not in data_generator.train_items[user]:
    #                 count[item] += 1
    #                 n += 1
    #             if n == 10:
    #                 break



    #     usersorted_id = []
    #     userbelong = []
    #     sorted_id = []
    #     belong = []
    #     with open('./curve/usersorted_id.txt','r') as f:
    #         usersorted_id=eval(f.read())
    #     with open('./curve/userbelong.txt','r') as f:
    #         userbelong=eval(f.read())
    #     with open('./curve/itembelong.txt','r') as f:
    #         belong=eval(f.read())
    #     with open('./curve/itemsorted_id.txt','r') as f:
    #         sorted_id=eval(f.read())

    #     count = count[sorted_id]
    #     x = list(range(6))
    #     y = [0,0,0,0,0,0]
    #     n_y = [0,0,0,0,0,0]
    #     for n, pop in enumerate(count):
    #         y[belong[n]] += pop
    #         n_y[belong[n]] += 1
    #     for i in range(6):
    #         y[i]/=1.0*n_y[i]

    #     with open('./curve/gcny_{}.txt'.format(args.loss), 'w') as f:
    #         f.write(str(y))
        

    #     if args.test == 'rubiboth':
    #         sig_yu, sig_yi = sess.run([model.sigmoid_yu, model.sigmoid_yi])

    #         sig_sum = [0,0,0,0,0,0]
    #         n_sig = [0,0,0,0,0,0]

    #         sig_yu = sig_yu[usersorted_id]
    #         for i, sig in enumerate(sig_yu):
    #             sig_sum[userbelong[i]] += sig
    #             n_sig[userbelong[i]] += 1
                
    #         for i in range(6):
    #             sig_sum[i]/=1.0*n_sig[i]
                
    #         with open('./curve/sig_yu_gcn.txt', 'w') as f:
    #             f.write(str(sig_sum))

    #         sig_sum = [0,0,0,0,0,0]
    #         n_sig = [0,0,0,0,0,0]

    #         sig_yi = sig_yi[sorted_id]
    #         for i, sig in enumerate(sig_yi):
    #             sig_sum[belong[i]] += sig
    #             n_sig[belong[i]] += 1
                
    #         for i in range(6):
    #             sig_sum[i]/=1.0*n_sig[i]
                
    #         with open('./curve/sig_yi_gcn.txt', 'w') as f:
    #             f.write(str(sig_sum))

    #         import matplotlib.pyplot as plt
    #         import matplotlib
    #         matplotlib.rcParams['figure.figsize'] = [10.5,6.5] # for square canvas
    #         matplotlib.rcParams['figure.subplot.left'] = 0.2
    #         matplotlib.rcParams['figure.subplot.bottom'] =0.1
    #         matplotlib.rcParams['figure.subplot.right'] = 0.8
    #         matplotlib.rcParams['figure.subplot.top'] = 0.8

    #         plt.switch_backend('agg')
    #         x = np.linspace(0, 60, 41)
    #         y = []
    #         for c in x:
    #             model.update_c(sess, c)
    #             test_users = list(data_generator.test_set.keys())
    #             ret = test(sess, model, test_users, method=args.test)
    #             y.append(ret['hr'][0])
    #         plt.plot(x, y, color='sandybrown')
    #         plt.scatter(x, y, color='sandybrown')
    #         plt.grid(alpha=0.3)
    #         plt.xlabel('c', size=24, fontweight='bold')
    #         plt.ylabel('HR@20', size=24, fontweight='bold')
    #         plt.xticks(size=16, fontweight='bold')
    #         plt.yticks(size = 16, fontweight='bold')

    #         plt.savefig('./curve/hr_addressa_causalgcn.png')
    #         plt.cla()



























    exit()

    epoch = 0
    best_epoch = 0
    epochs_best_result = dict()
    epochs_best_result['recall'] = 0
    epochs_best_result['hit_ratio'] = 0
    epochs_best_result['ndcg'] = 0
    best_epoch_c = 0
    while True:
        best_c = 0
        epoch_best_result = dict()
        epoch_best_result['recall'] = 0
        epoch_best_result['hit_ratio'] = 0
        epoch_best_result['ndcg'] = 0
        epoch += args.log_interval
        try:
            model_file = weights_save_path + 'weights_{}-{}'.format(args.saveID, epoch)
            saver.restore(sess, model_file)
        except ValueError:
            break
        print(epoch, ':restored.')
        users_to_test = list(data_generator.test_set.keys())
        msg = ''
        base = args.base
        for c in range(21):
            c_v = base + c/10
            model.update_c(sess, c_v)
            ret = test(sess, model, users_to_test, drop_flag=True,method="causal_c")
            perf_str = 'C=[%s], recall=[%s], ' \
                        'hr=[%s], ndcg=[%s]' % \
                        (c_v, 
                            ', '.join(['%.5f' % r for r in ret['recall']]),
                            ', '.join(['%.5f' % r for r in ret['hr']]),
                            ', '.join(['%.5f' % r for r in ret['ndcg']]))
            if ret['hr'][1] > epoch_best_result['hit_ratio']:
                best_c = c_v
                epoch_best_result['recall'] = ret['recall'][1]
                epoch_best_result['hit_ratio'] = ret['hr'][1]
                epoch_best_result['ndcg'] = ret['ndcg'][1]
            # print(perf_str)
            msg += perf_str + '\n'
        base = best_c - 0.1
        for c in range(21):
            c_v = base + c/100
            model.update_c(sess, c_v)
            ret = test(sess, model, users_to_test, drop_flag=True,method="causal_c")
            perf_str = 'C=[%s], recall=[%s], ' \
                        'hr=[%s], ndcg=[%s]' % \
                        (c_v, 
                            ', '.join(['%.5f' % r for r in ret['recall']]),
                            ', '.join(['%.5f' % r for r in ret['hr']]),
                            ', '.join(['%.5f' % r for r in ret['ndcg']]))
            if ret['hr'][1] > epoch_best_result['hit_ratio']:
                best_c = c_v
                epoch_best_result['recall'] = ret['recall'][1]
                epoch_best_result['hit_ratio'] = ret['hr'][1]
                epoch_best_result['ndcg'] = ret['ndcg'][1]
            # print(perf_str)
            msg += perf_str + '\n'
        msg += ('best c = %.2f, recall@20=%.5f,\nhit@20=%.5f,\nndcg@20=%.5f' % (best_c,
                        epoch_best_result['recall'],
                        epoch_best_result['hit_ratio'],
                        epoch_best_result['ndcg']))
        if os.path.exists('check_c/') == False:
            os.makedirs('check_c/')
        with open('check_c/{}_{}_{}_epoch_{}.txt'.format(args.model_type, args.dataset, args.saveID, epoch), 'w') as f:
            f.write(msg)
        if epoch_best_result['hit_ratio'] > epochs_best_result['hit_ratio']:
            best_epoch = epoch
            best_epoch_c = best_c
            epochs_best_result['recall'] = epoch_best_result['recall']
            epochs_best_result['hit_ratio'] = epoch_best_result['hit_ratio']
            epochs_best_result['ndcg'] = epoch_best_result['ndcg']
    print('best epoch = %d, best c = %.2f, recall@20=%.5f,\nhit@20=%.5f,\nndcg@20=%.5f' % (best_epoch, best_epoch_c,
                epochs_best_result['recall'],
                epochs_best_result['hit_ratio'],
                epochs_best_result['ndcg']))



        # save_path = '%soutput/%s/%s.result' % (args.proj_path, args.dataset, model.model_type)
        # ensureDir(save_path)
        # f = open(save_path, 'a')

        # f.write(
        #     'embed_size=%d, lr=%.4f, layer_size=%s, node_dropout=%s, mess_dropout=%s, regs=%s, adj_type=%s\n\t%s\n'
        #     % (args.embed_size, args.lr, args.layer_size, args.node_dropout, args.mess_dropout, args.regs,
        #     args.adj_type, final_perf))
        # f.close()