import tensorflow as tf
import numpy as np
import os
import sys
import random
import collections
from parse import parse_args
import math
from cyutils.sampler import CyCPRSampler, CyPairNegSampler, CyDICENegSampler
import scipy.sparse as sp
import heapq
import math
import time
from tqdm import tqdm

from cyutils.evaluator import CyEvaluator

def create_norm_adj(u_interacts, i_interacts, n_user, n_item):
    """Create normalized adjacency matrix.

    Returns:
        sp.csr_matrix: Normalized adjacency matrix.
        
    """

    # Create interaction matrix.
    R = sp.coo_matrix(
        ([1.] * len(u_interacts), (u_interacts, i_interacts)),
        shape=(n_user, n_item),
        dtype=np.float32).tocsr()

    # Create adjacency matrix.
    zero_u_mat = sp.csr_matrix((n_user, n_user), dtype=np.float32)
    zero_i_mat = sp.csr_matrix((n_item, n_item), dtype=np.float32)
    adj = sp.hstack([
        sp.vstack([zero_u_mat, R.T]),
        sp.vstack([R, zero_i_mat])
    ]).tocsr()

    D = np.array(adj.sum(axis=1))
    # Normalize adjacency matrix.
    row_sum = D.ravel()
    # Symmetric normalized Laplacian
    s_diag_flat = np.power(row_sum,
                           -0.5,
                           out=np.zeros_like(row_sum),
                           where=row_sum != 0)
    s_diag = sp.diags(s_diag_flat)
    s_norm_adj = s_diag.dot(adj).dot(s_diag)

    return s_norm_adj


def create_ngcf_embed(all_embeds_0, s_norm_adj, n_layer, W1s, b1s, W2s, b2s,
                      args):
    s_norm_adj = sp_mat_2_sp_tensor(s_norm_adj)
    ego_embeds = all_embeds_0
    all_embeds = [ego_embeds]

    for i in range(n_layer):
        neigh_embeds = tf.sparse_tensor_dense_matmul(s_norm_adj, ego_embeds)
        sum_embeds = tf.nn.leaky_relu(tf.matmul(neigh_embeds, W1s[i]) + b1s[i])
        bi_embeds = tf.nn.leaky_relu(
            tf.matmul(tf.multiply(ego_embeds, neigh_embeds), W2s[i]) + b2s[i])
        ego_embeds = sum_embeds + bi_embeds
        all_embeds += [tf.nn.l2_normalize(ego_embeds, axis=1)]

    all_embeds = tf.concat(all_embeds, 1)

    return all_embeds

def sp_mat_2_sp_tensor(X):
    """Convert a scipy sparse matrix to tf.SparseTensor.

    Returns:
        tf.SparseTensor: SparseTensor after conversion.
        
    """
    coo = X.tocoo().astype(np.float32)
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)

def create_lightgcn_embed(all_embeds_0, s_norm_adj, n_layer):
    s_norm_adj = sp_mat_2_sp_tensor(s_norm_adj)
    ego_embeds = all_embeds_0
    all_embeds = [ego_embeds]

    for _ in range(n_layer):
        ego_embeds = tf.sparse_tensor_dense_matmul(s_norm_adj, ego_embeds)
        all_embeds += [ego_embeds]

    all_embeds = tf.stack(all_embeds, 1)
    all_embeds = tf.reduce_mean(all_embeds, axis=1, keepdims=False)

    return all_embeds



class Evaluator(CyEvaluator):
    def __init__(
        self,
        dataset,
        eval_type,
        metrics=["Recall"],
        ks=[20],
        n_thread=8,
    ):
        self.eval_type = eval_type
        self.metrics = metrics
        self.ks = ks
        self.i_degrees = dataset.i_degrees

        evalset = dataset.evalsets[eval_type]
        self.eval_users = np.array(sorted(evalset.keys()))

        self.n_i_group = 1
        self.n_u_group = 1
        self.u_groups = None
        self.i_groups = None

        # (n_user, n_group_metric, n_k, n_i_group)
        self.metric_values_all_u = np.empty([
            len(self.eval_users),
            len(self.metrics),
            len(ks), 1 if self.n_i_group == 1 else self.n_i_group + 1
        ],
                                            dtype=np.float32)

        evalset_list = [evalset[u] for u in self.eval_users]
        super().__init__(evalset_list,
                         [m.encode("utf_8") for m in self.metrics], ks,
                         n_thread, self.n_i_group, self.i_groups,
                         self.i_degrees, self.metric_values_all_u)

    def update(self, batch_ratings, batch_users_idx):
        self.eval(batch_ratings, np.asarray(batch_users_idx, dtype=np.int32))

    def update_top_k(self, batch_top_k, batch_users_idx):
        self.eval_top_k(batch_top_k, np.asarray(batch_users_idx,
                                                dtype=np.int32))

    def update_final(self):
        self.metric_values_group_i = np.nanmean(self.metric_values_all_u,
                                                axis=0)
        if self.n_u_group > 1:
            # (n_metric, n_k, n_user)
            u_values = self.metric_values_all_u[:, :, :, 0].transpose(1, 2, 0)

            self.metric_values_group_u = np.stack(
                [np.nanmean(u_values, axis=-1)] + [
                    np.nanmean(u_values[:, :, self.u_groups == i], axis=-1)
                    for i in range(self.n_u_group)
                ],
                axis=-1)

    def get(self, metric, k, group=-1):
        try:
            k_idx = self.ks.index(k)
        except:
            raise ValueError("{} is not in ks.".format(k))
        try:
            m_idx = self.metrics.index(metric)
        except:
            raise ValueError("{} is not in metrics.".format(metric))
        # (n_metric, n_k, n_i_group)
        return self.metric_values_group_i[m_idx, k_idx, group + 1]

    def _prep_lines(self, metrics, values):
        lines = []
        for metric, result_m in zip(self.metrics, values):
            if metric not in metrics:
                continue
            for k, result_m_k in zip(self.ks, result_m):
                lines.append("{:<10}@{:<3}:".format(metric, k) + (
                    "{:10.5f}".format(result_m_k) if isinstance(
                        result_m_k, np.float32) else "{}".format(' '.join(
                            "{:10.5f}".format(x) for x in result_m_k))))
        return lines

    def __str__(self):
        lines = []
        lines.append("[ {} set ]".format(self.eval_type))
        lines.append("---- Item ----")
        lines.extend(self._prep_lines(self.metrics,
                                      self.metric_values_group_i))
        if self.n_u_group > 1:
            lines.append("---- User ----")
            lines.extend(
                self._prep_lines([m for m in self.metrics if m != "Rec"],
                                 self.metric_values_group_u))
        return '\n'.join(lines)


def create_evaluators(dataset, eval_types, metrics, ks, n_thread):
    evaluators = {}
    for eval_type in eval_types:
        evaluators[eval_type] = Evaluator(dataset, eval_type, metrics, ks, n_thread)
    return evaluators

def l2_embed_loss(*args):
    loss = 0
    for embeds in args:
        loss += tf.reduce_sum(tf.square(embeds), axis=1)
    return tf.reduce_mean(loss)

def discrepency_loss(x, y):
    # dcor has numerical problem when implemented by tensorflow
    # (loss would become nan),
    # so we use l2 loss instead.
    # return dcor(x, y)
    return tf.reduce_mean(tf.square(x - y))

def mask_bpr_loss(pos_scores, neg_scores, mask):
    loss = -tf.reduce_mean(mask * tf.math.log_sigmoid(pos_scores - neg_scores))
    return loss

def bpr_loss(pos_scores, neg_scores):
    pairwise_obj = pos_scores - neg_scores
    loss = tf.reduce_mean(tf.math.softplus(-pairwise_obj))
    return loss

def inner_product(u_embeds, i_embeds):
    output = tf.reduce_sum(u_embeds * i_embeds, axis=1)
    return output



def batch_iterator(data, batch_size, drop_last=False):
    """Generate batches.

    Args:
        data (list or numpy.ndarray): Input data.
        batch_size (int): Size of each batch except for the last one.
    """
    length = len(data)
    if drop_last:
        n_batch = length // batch_size
    else:
        n_batch = math.ceil(length / batch_size)
    for i in range(n_batch):
        yield data[i * batch_size:(i + 1) * batch_size]



# def create_norm_adj(u_interacts, i_interacts, n_user, n_item):
#     """Create normalized adjacency matrix.

#     Returns:
#         sp.csr_matrix: Normalized adjacency matrix.
        
#     """

#     # Create interaction matrix.
#     R = sp.coo_matrix(
#         ([1.] * len(u_interacts), (u_interacts, i_interacts)),
#         shape=(n_user, n_item),
#         dtype=np.float32).tocsr()

#     # Create adjacency matrix.
#     zero_u_mat = sp.csr_matrix((n_user, n_user), dtype=np.float32)
#     zero_i_mat = sp.csr_matrix((n_item, n_item), dtype=np.float32)
#     adj = sp.hstack([
#         sp.vstack([zero_u_mat, R.T]),
#         sp.vstack([R, zero_i_mat])
#     ]).tocsr()

#     D = np.array(adj.sum(axis=1))
#     # Normalize adjacency matrix.
#     row_sum = D.ravel()
#     # Symmetric normalized Laplacian
#     s_diag_flat = np.power(row_sum,
#                            -0.5,
#                            out=np.zeros_like(row_sum),
#                            where=row_sum != 0)
#     s_diag = sp.diags(s_diag_flat)
#     s_norm_adj = s_diag.dot(adj).dot(s_diag)

#     return s_norm_adj


class DICESampler(object):
    def __init__(self, dataset, args):
        self.batch_size = args.batch_size
        self.n_thread = 10

        self.n_user = dataset.n_users
        self.n_item = dataset.n_items
        self.n_interact = dataset.n_interact
        self.train = dataset.train
        self.u_interacts = dataset.u_interacts
        self.i_interacts = dataset.i_interacts
        self.i_degrees = dataset.i_degrees
        self.n_step = dataset.n_interact // self.batch_size
        self.sample_size = self.n_step * self.batch_size

        self.margin = args.margin
        self.min_size = args.min_size

        self.neg_items = np.empty(self.sample_size, dtype=np.int32)
        self.neg_mask = np.empty(self.sample_size, dtype=np.float32)
        self.sampler = CyDICENegSampler(self.train, self.n_item, self.min_size,
                                        self.i_degrees, self.neg_items,
                                        self.neg_mask, self.n_thread)
        

    def sample(self):
        idx = np.random.choice(self.n_interact, size=self.sample_size)
        users = self.u_interacts[idx]
        pos_items = self.i_interacts[idx]
        rand = np.random.rand(self.sample_size * 2).astype(np.float32)
        self.sampler.sample(users, pos_items, rand, self.margin)
        return zip(batch_iterator(users, batch_size=self.batch_size),
                   batch_iterator(pos_items, batch_size=self.batch_size),
                   batch_iterator(self.neg_items, batch_size=self.batch_size),
                   batch_iterator(self.neg_mask, batch_size=self.batch_size))


class DICE(object):
    """LightGCN model

    SIGIR 2020. He, Xiangnan, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, and Meng Wang.
    "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." arXiv
    preprint arXiv:2002.02126 (2020).
    """
    def __init__(self, args, dataset):
        """Initializing the model. Create parameters, placeholders, embeddings and loss function.

        """
        self.dataset = dataset
        self._build_graph(args)
        self.saver = tf.train.Saver(max_to_keep=1)
        self.evaluators = create_evaluators(self.dataset, ['valid','test','test_id'],
                                                ["Recall", "Precision", "NDCG", "Rec", "ARP", "Hits"],[20],8)

    def _build_graph(self, args):
        self.initializer = tf.contrib.layers.xavier_initializer()
        self.create_variables(args)
        self.create_embeds(args)
        self.create_batch_ratings(args)
        self.sampler = DICESampler(self.dataset, args)
        self.int_weight = tf.placeholder(tf.float32, shape=())
        self.pop_weight = tf.placeholder(tf.float32, shape=())
        self.dis_pen = args.dis_pen
        self.create_loss(args)
        self.opt = tf.train.AdamOptimizer(learning_rate=args.lr).minimize(
            self.loss)
        self.int_weight_v = args.int_weight
        self.pop_weight_v = args.pop_weight

    def create_variables(self, args):
        self.all_embeds_0 = tf.get_variable(
            "all_embeds_0",
            shape=[self.dataset.n_users + self.dataset.n_items, args.embed_size],
            initializer=self.initializer)
        self.u_embeds_0, self.i_embeds_0 = tf.split(
            self.all_embeds_0, [self.dataset.n_users, self.dataset.n_items], 0)
        self.int_embeds_0, self.pop_embeds_0 = tf.split(
            self.all_embeds_0, 2, 1)

        embed_size = args.embed_size // 2
        if args.embed_type == "ngcf":
            self.int_W1s = []
            self.int_b1s = []
            self.int_W2s = []
            self.int_b2s = []
            self.pop_W1s = []
            self.pop_b1s = []
            self.pop_W2s = []
            self.pop_b2s = []
            for i in range(args.n_layer):
                self.int_W1s.append(
                    tf.get_variable("int_W1_{}".format(i),
                                    shape=[embed_size, embed_size],
                                    initializer=self.initializer))
                self.int_b1s.append(
                    tf.get_variable("int_b1_{}".format(i),
                                    shape=[1, embed_size],
                                    initializer=self.initializer))
                self.int_W2s.append(
                    tf.get_variable("int_W2_{}".format(i),
                                    shape=[embed_size, embed_size],
                                    initializer=self.initializer))
                self.int_b2s.append(
                    tf.get_variable("int_b2_{}".format(i),
                                    shape=[1, embed_size],
                                    initializer=self.initializer))
                self.pop_W1s.append(
                    tf.get_variable("pop_W1_{}".format(i),
                                    shape=[embed_size, embed_size],
                                    initializer=self.initializer))
                self.pop_b1s.append(
                    tf.get_variable("pop_b1_{}".format(i),
                                    shape=[1, embed_size],
                                    initializer=self.initializer))
                self.pop_W2s.append(
                    tf.get_variable("pop_W2_{}".format(i),
                                    shape=[embed_size, embed_size],
                                    initializer=self.initializer))
                self.pop_b2s.append(
                    tf.get_variable("pop_b2_{}".format(i),
                                    shape=[1, embed_size],
                                    initializer=self.initializer))

        if args.inference_type == "mlp":
            weight_sizes = [2 * embed_size
                            ] + [x // 2 for x in args.weight_sizes]
            self.int_Ws = []
            self.int_bs = []
            self.pop_Ws = []
            self.pop_bs = []
            for i in range(len(args.weight_sizes)):
                self.int_Ws.append(
                    tf.get_variable(
                        "int_W_{}".format(i),
                        shape=[weight_sizes[i], weight_sizes[i + 1]],
                        initializer=self.initializer))
                self.int_bs.append(
                    tf.get_variable("int_b_{}".format(i),
                                    shape=[1, weight_sizes[i + 1]],
                                    initializer=self.initializer))
            self.int_h = tf.get_variable(
                "int_h",
                shape=[weight_sizes[-1] + embed_size, 1],
                initializer=self.initializer)
            for i in range(len(args.weight_sizes)):
                self.pop_Ws.append(
                    tf.get_variable(
                        "pop_W_{}".format(i),
                        shape=[weight_sizes[i], weight_sizes[i + 1]],
                        initializer=self.initializer))
                self.pop_bs.append(
                    tf.get_variable("pop_b_{}".format(i),
                                    shape=[1, weight_sizes[i + 1]],
                                    initializer=self.initializer))
            self.pop_h = tf.get_variable(
                "pop_h",
                shape=[weight_sizes[-1] + embed_size, 1],
                initializer=self.initializer)

    def create_embeds(self, args):
        print(self.dataset.n_users)
        s_norm_adj = create_norm_adj(self.dataset.u_interacts,
                                     self.dataset.i_interacts,
                                     self.dataset.n_users, self.dataset.n_items)
        if args.embed_type == "ngcf":
            self.int_embeds = create_ngcf_embed(self.int_embeds_0, s_norm_adj,
                                                args.n_layer, self.int_W1s,
                                                self.int_b1s, self.int_W2s,
                                                self.int_b2s, args)
            self.pop_embeds = create_ngcf_embed(self.pop_embeds_0, s_norm_adj,
                                                args.n_layer, self.pop_W1s,
                                                self.pop_b1s, self.pop_W2s,
                                                self.pop_b2s, args)
        elif args.embed_type == "lightgcn":
            self.int_embeds = create_lightgcn_embed(self.int_embeds_0,
                                                    s_norm_adj, args.n_layer)
            self.pop_embeds = create_lightgcn_embed(self.pop_embeds_0,
                                                    s_norm_adj, args.n_layer)
        self.all_embeds = tf.concat([self.int_embeds, self.pop_embeds], 1)
        self.u_embeds, self.i_embeds = tf.split(
            self.all_embeds, [self.dataset.n_users, self.dataset.n_items], 0)

    def create_loss(self, args):
        self.batch_pos_i = tf.placeholder(tf.int32, shape=(None, ))
        self.batch_neg_i = tf.placeholder(tf.int32, shape=(None, ))
        self.batch_neg_mask = tf.placeholder(tf.float32, shape=(None, ))
        batch_pos_i_embeds = tf.nn.embedding_lookup(self.i_embeds,
                                                    self.batch_pos_i)
        batch_neg_i_embeds = tf.nn.embedding_lookup(self.i_embeds,
                                                    self.batch_neg_i)
        users_int, users_pop = tf.split(self.batch_u_embeds, 2, 1)
        items_p_int, items_p_pop = tf.split(batch_pos_i_embeds, 2, 1)
        items_n_int, items_n_pop = tf.split(batch_neg_i_embeds, 2, 1)
        if args.inference_type == "inner_product":
            p_score_int = inner_product(users_int, items_p_int)
            n_score_int = inner_product(users_int, items_n_int)
            p_score_pop = inner_product(users_pop, items_p_pop)
            n_score_pop = inner_product(users_pop, items_n_pop)
        elif args.inference_type == "mlp":
            p_score_int = mlp(users_int, items_p_int, self.int_Ws, self.int_bs,
                              self.int_h, args)
            n_score_int = mlp(users_int, items_n_int, self.int_Ws, self.int_bs,
                              self.int_h, args)
            p_score_pop = mlp(users_pop, items_p_pop, self.pop_Ws, self.pop_bs,
                              self.pop_h, args)
            n_score_pop = mlp(users_pop, items_n_pop, self.pop_Ws, self.pop_bs,
                              self.pop_h, args)

        p_score_total = p_score_int + p_score_pop
        n_score_total = n_score_int + n_score_pop

        self.loss_int = mask_bpr_loss(p_score_int, n_score_int,
                                      self.batch_neg_mask)
        self.loss_pop = mask_bpr_loss(
            n_score_pop, p_score_pop, self.batch_neg_mask) + mask_bpr_loss(
                p_score_pop, n_score_pop, 1 - self.batch_neg_mask)
        self.loss_total = bpr_loss(p_score_total, n_score_total)

        user_int = tf.concat([users_int, users_int], 0)
        user_pop = tf.concat([users_pop, users_pop], 0)
        item_int = tf.concat([items_p_int, items_n_int], 0)
        item_pop = tf.concat([items_p_pop, items_n_pop], 0)
        self.discrepency_loss = discrepency_loss(
            item_int, item_pop) + discrepency_loss(user_int, user_pop)
        self.mf_loss = self.int_weight * self.loss_int + self.pop_weight * self.loss_pop + self.loss_total - self.dis_pen * self.discrepency_loss

        self.reg_loss = args.reg * l2_embed_loss(self.all_embeds)
        if args.embed_type == "ngcf":
            for x in self.int_W1s + self.int_b1s + self.int_W2s + self.int_b2s + self.pop_W1s + self.pop_b1s + self.pop_W2s + self.pop_b2s:
                self.reg_loss += args.weight_reg * tf.nn.l2_loss(x)
        if args.inference_type == "mlp":
            for x in self.int_Ws + self.int_bs + [
                    self.int_h
            ] + self.pop_Ws + self.pop_bs + [self.pop_h]:
                self.reg_loss += args.weight_reg * tf.nn.l2_loss(x)

        self.loss = self.mf_loss + self.reg_loss

    def create_batch_ratings(self, args):
        self.batch_u = tf.placeholder(tf.int32, shape=(None, ))
        self.batch_u_embeds = tf.nn.embedding_lookup(self.u_embeds,
                                                     self.batch_u)
        users_int, _ = tf.split(self.batch_u_embeds, 2, 1)
        i_int_embeds, _ = tf.split(self.i_embeds, 2, 1)

        if args.inference_type == "inner_product":
            self.batch_ratings = tf.matmul(users_int,
                                           i_int_embeds,
                                           transpose_b=True)
        elif args.inference_type == "mlp":
            u_size = tf.shape(users_int)[0]
            i_size = tf.shape(i_int_embeds)[0]
            u_repeats = tf.repeat(users_int, i_size, axis=0)
            i_tiles = tf.tile(i_int_embeds, [u_size, 1])
            scores = mlp(u_repeats, i_tiles, self.int_Ws, self.int_bs,
                         self.int_h, args)
            self.batch_ratings = tf.reshape(scores, [u_size, i_size])

    def train_1_epoch(self, args):
        #self.timer.start("Epoch {}".format(epoch))
        losses = []
        mf_losses = []
        dis_losses = []
        reg_losses = []
        for users, pos_items, neg_items, neg_mask in tqdm(self.sampler.sample()):
            _, batch_loss, batch_mf_loss, batch_dis_loss, batch_reg_loss = self.sess.run(
                [
                    self.opt, self.loss, self.mf_loss, self.discrepency_loss,
                    self.reg_loss
                ],
                feed_dict={
                    self.int_weight: self.int_weight_v,
                    self.pop_weight: self.pop_weight_v,
                    self.batch_u: users,
                    self.batch_pos_i: pos_items,
                    self.batch_neg_i: neg_items,
                    self.batch_neg_mask: neg_mask
                },
            )
            losses.append(batch_loss)
            mf_losses.append(batch_mf_loss)
            dis_losses.append(batch_dis_loss)
            reg_losses.append(batch_reg_loss)
        

        self.int_weight_v *= args.loss_decay
        self.pop_weight_v *= args.loss_decay
        self.sampler.margin *= args.margin_decay

        print(
            "int_weight = {:.5f}, pop_weight = {:.5f}, margin = {:.5f}".format(
                self.int_weight_v, self.pop_weight_v, self.sampler.margin))
        # self.timer.stop(
        #     "loss = {:.5f} = {:.5f} (dis_loss = {:.5f}) + {:.5f}".format(
        #         np.mean(losses), np.mean(mf_losses), np.mean(dis_losses),
        #         np.mean(reg_losses)))
        perf_str="loss = {:.5f} = {:.5f} (dis_loss = {:.5f}) + {:.5f}".format(np.mean(losses), np.mean(mf_losses), np.mean(dis_losses),np.mean(reg_losses))
        return perf_str

    def eval(self, args):
        self.timer.start("Evaluation")
        perf_str=""
        for evaluator in self.evaluators.values():
            for idx, batch_u in enumerate(
                    batch_iterator(evaluator.eval_users,
                                   args.eval_batch_size)):
                batch_users_idx = range(
                    idx * args.eval_batch_size,
                    idx * args.eval_batch_size + len(batch_u))
                batch_ratings = self.sess.run(
                    self.batch_ratings, feed_dict={self.batch_u: batch_u})

                for idx, user in enumerate(batch_u):
                    batch_ratings[idx][self.dataset.train[user]] = -np.inf

                evaluator.update(batch_ratings, batch_users_idx)

            evaluator.update_final()
            print(evaluator)
            perf_str+=evaluator.__str__()

        self.timer.stop()
        return perf_str

    def predict(self,users,items=None):
        if items==None:
            items = list(range(self.dataset.n_items))
        rate_batch = self.sess.run(
                    self.batch_ratings, feed_dict={self.batch_u: users})
        return rate_batch



    def add_sess(self,sess):
        self.sess=sess

    
    def eval(self, args):
        perf_str=""
        for evaluator in self.evaluators.values():
            for idx, batch_u in enumerate(
                    batch_iterator(evaluator.eval_users,
                                   args.batch_size)):
                batch_users_idx = range(
                    idx * args.batch_size,
                    idx * args.batch_size + len(batch_u))
                batch_ratings = self.sess.run(
                    self.batch_ratings, feed_dict={self.batch_u: batch_u})

                for idx, user in enumerate(batch_u):
                    batch_ratings[idx][self.dataset.train[user]] = -np.inf

                evaluator.update(batch_ratings, batch_users_idx)

            evaluator.update_final()
            perf_str+=evaluator.__str__()
        return perf_str
    
    


class BPRMF:
    def __init__(self, args, data_config):
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.decay = args.regs
        self.emb_dim = args.embed_size
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.verbose = args.verbose
        self.c = args.c
        self.alpha = args.alpha
        self.beta = args.beta        
        #placeholders
        self.users = tf.placeholder(tf.int32, shape = (None,))
        self.pos_items = tf.placeholder(tf.int32, shape = (None,))
        self.neg_items = tf.placeholder(tf.int32, shape = (None,))

        #initiative weights
        self.weights = self.init_weights()

        #neting
        user_embedding = tf.nn.embedding_lookup(self.weights['user_embedding'], self.users)
        pos_item_embedding = tf.nn.embedding_lookup(self.weights['item_embedding'], self.pos_items)
        neg_item_embedding = tf.nn.embedding_lookup(self.weights['item_embedding'], self.neg_items)
        user_rand_embedding = tf.nn.embedding_lookup(self.weights['user_rand_embedding'], self.users)
        item_rand_embedding = tf.nn.embedding_lookup(self.weights['item_rand_embedding'], self.pos_items)
        

        self.const_embedding = self.weights['c']
        self.user_c = tf.nn.embedding_lookup(self.weights['user_c'], self.users)

        self.batch_ratings = tf.matmul(user_embedding, pos_item_embedding, transpose_a=False, transpose_b = True)    #prediction, shape(user_embedding) != shape(pos_item_embedding)
        self.user_const_ratings = self.batch_ratings - tf.matmul(self.const_embedding, pos_item_embedding, transpose_a=False, transpose_b = True)   #auto tile
        self.item_const_ratings = self.batch_ratings - tf.matmul(user_embedding, self.const_embedding, transpose_a=False, transpose_b = True)       #auto tile
        self.user_rand_ratings = self.batch_ratings - tf.matmul(user_rand_embedding, pos_item_embedding, transpose_a=False, transpose_b = True)
        self.item_rand_ratings = self.batch_ratings - tf.matmul(user_embedding, item_rand_embedding, transpose_a=False, transpose_b = True)


        self.mf_loss, self.reg_loss = self.create_bpr_loss(user_embedding, pos_item_embedding, neg_item_embedding)
        self.loss = self.mf_loss + self.reg_loss

        # self.opt = tf.train.RMSPropOptimizer(learning_rate = self.lr).minimize(self.loss)
        trainable_v1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'parameter')
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss, var_list = trainable_v1)
        # two branch
        self.w = tf.Variable(self.initializer([self.emb_dim,1]), name = 'item_branch')
        self.w_user = tf.Variable(self.initializer([self.emb_dim,1]), name = 'user_branch')
        self.sigmoid_yu = tf.squeeze(tf.nn.sigmoid(tf.matmul(self.weights['user_embedding'], self.w_user)))
        self.sigmoid_yi = tf.squeeze(tf.nn.sigmoid(tf.matmul(self.weights['item_embedding'], self.w)))
        # two branch bpr
        self.mf_loss_two, self.reg_loss_two = self.create_bpr_loss_two_brach(user_embedding, pos_item_embedding, neg_item_embedding)
        self.loss_two = self.mf_loss_two + self.reg_loss_two
        self.opt_two = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_two)
        # two branch bce
        self.mf_loss_two_bce, self.reg_loss_two_bce = self.create_bce_loss_two_brach(user_embedding, pos_item_embedding, neg_item_embedding)
        self.loss_two_bce = self.mf_loss_two_bce + self.reg_loss_two_bce
        self.opt_two_bce = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_two_bce)
        # two branch bce user&item
        self.mf_loss_two_bce_both, self.reg_loss_two_bce_both = self.create_bce_loss_two_brach_both(user_embedding, pos_item_embedding, neg_item_embedding)
        self.loss_two_bce_both = self.mf_loss_two_bce_both + self.reg_loss_two_bce_both
        self.opt_two_bce_both = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_two_bce_both)
        # 2-stage training
        self.mf_loss2, self.reg_loss2 = self.create_bpr_loss2(user_embedding, self.const_embedding, pos_item_embedding, neg_item_embedding)
        self.loss2 = self.mf_loss2 + self.reg_loss2
        trainable_v2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'const_embedding')
        self.opt2 = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss2, var_list = trainable_v2)


        self.mf_loss2_bce, self.reg_loss2_bce = self.create_bce_loss2(user_embedding, self.const_embedding, pos_item_embedding, neg_item_embedding)
        self.loss2_bce = self.mf_loss2_bce + self.reg_loss2_bce
        self.opt2_bce = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss2_bce, var_list = trainable_v2)
        # self.opt = tf.train.RMSPropOptimizer(learning_rate = self.lr).minimize(self.loss)
        
        
        self.opt3 = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss2, var_list = [self.weights['user_embedding'],self.weights['item_embedding']])
        self.opt3_bce = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss2_bce, var_list = [self.weights['user_embedding'],self.weights['item_embedding']])

        self._statistics_params()

        self.mf_loss_bce, self.reg_loss_bce = self.create_bce_loss(user_embedding, pos_item_embedding, neg_item_embedding)
        self.loss_bce = self.mf_loss_bce + self.reg_loss_bce
        self.opt_bce = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_bce)


        # user wise two branch mf
        self.mf_loss_userc_bce, self.reg_loss_userc_bce = self.create_bce_loss_userc(user_embedding, pos_item_embedding, neg_item_embedding)
        self.loss_userc_bce = self.mf_loss_userc_bce + self.reg_loss_userc_bce
        self.opt_userc_bce = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_userc_bce, var_list = [self.weights['user_c']])
        # self.opt = tf.train.RMSPropOptimizer(learning_rate = self.lr).minimize(self.loss)




    def init_weights(self):
        weights = dict()
        self.initializer = tf.contrib.layers.xavier_initializer()
        initializer = self.initializer
        with tf.variable_scope('parameter'):
            weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name = 'user_embedding')
            weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name = 'item_embedding')
            weights['user_rand_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name = 'user_rand_embedding', trainable = False)
            weights['item_rand_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name = 'item_rand_embedding', trainable = False)
        with tf.variable_scope('const_embedding'):
            self.rubi_c = tf.Variable(tf.zeros([1]), name = 'rubi_c')
            weights['c'] = tf.Variable(tf.zeros([1, self.emb_dim]), name = 'c')
        
        weights['user_c'] = tf.Variable(tf.zeros([self.n_users, 1]), name = 'user_c_v')

        return weights

    def create_bpr_loss_two_brach(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)   #users, pos_items, neg_items have the same shape
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        # item stop


        # pos_items_stop = tf.stop_gradient(pos_items)
        # neg_items_stop = tf.stop_gradient(neg_items)
        pos_items_stop = pos_items
        neg_items_stop = neg_items

        self.pos_item_scores = tf.matmul(pos_items_stop,self.w)
        self.neg_item_scores = tf.matmul(neg_items_stop,self.w)
        # first branch
        pos_scores = pos_scores*tf.nn.sigmoid(self.pos_item_scores)
        neg_scores = neg_scores*tf.nn.sigmoid(self.neg_item_scores)
        maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))
        self.rubi_ratings = (self.batch_ratings-self.rubi_c)*tf.squeeze(tf.nn.sigmoid(self.pos_item_scores))
        self.direct_minus_ratings = self.batch_ratings-self.rubi_c*tf.squeeze(tf.nn.sigmoid(self.pos_item_scores))
        
        self.mf_loss_ori_bce = tf.negative(tf.reduce_mean(maxi))
        # second branch
        maxi_item = tf.log(tf.nn.sigmoid(self.pos_item_scores - self.neg_item_scores))
        self.mf_loss_item_bce = tf.negative(tf.reduce_mean(maxi_item))
        # unify
        mf_loss = self.mf_loss_ori_bce + self.alpha*self.mf_loss_item_bce
        # regular
        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer/self.batch_size

        
        reg_loss = self.decay * regularizer
        return mf_loss, reg_loss

    def create_bce_loss_two_brach(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)   #users, pos_items, neg_items have the same shape
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        # item score
        # pos_items_stop = tf.stop_gradient(pos_items)
        # neg_items_stop = tf.stop_gradient(neg_items)
        pos_items_stop = pos_items
        neg_items_stop = neg_items
        self.pos_item_scores = tf.matmul(pos_items_stop,self.w)
        self.neg_item_scores = tf.matmul(neg_items_stop,self.w)
        # self.rubi_ratings = (self.batch_ratings-self.rubi_c)*tf.squeeze(tf.nn.sigmoid(self.pos_item_scores))
        # self.direct_minus_ratings = self.batch_ratings-self.rubi_c*tf.squeeze(tf.nn.sigmoid(self.pos_item_scores))
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
        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer/self.batch_size
        reg_loss = self.decay * regularizer
        return mf_loss, reg_loss

    def create_bce_loss_two_brach_both(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)   #users, pos_items, neg_items have the same shape
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        # item score
        # pos_items_stop = tf.stop_gradient(pos_items)
        # neg_items_stop = tf.stop_gradient(neg_items)
        pos_items_stop = pos_items
        neg_items_stop = neg_items
        users_stop = users
        self.pos_item_scores = tf.matmul(pos_items_stop,self.w)
        self.neg_item_scores = tf.matmul(neg_items_stop,self.w)
        self.user_scores = tf.matmul(users_stop, self.w_user)
        # self.rubi_ratings_both = (self.batch_ratings-self.rubi_c)*(tf.transpose(tf.nn.sigmoid(self.pos_item_scores))+tf.nn.sigmoid(self.user_scores))
        # self.direct_minus_ratings_both = self.batch_ratings-self.rubi_c*(tf.transpose(tf.nn.sigmoid(self.pos_item_scores))+tf.nn.sigmoid(self.user_scores))
        self.rubi_ratings_both = (self.batch_ratings-self.rubi_c)*tf.transpose(tf.nn.sigmoid(self.pos_item_scores))*tf.nn.sigmoid(self.user_scores)
        self.rubi_ratings_both_poptest = self.batch_ratings*tf.nn.sigmoid(self.user_scores)
        self.direct_minus_ratings_both = self.batch_ratings-self.rubi_c*tf.transpose(tf.nn.sigmoid(self.pos_item_scores))*tf.nn.sigmoid(self.user_scores)
        # first branch
        # fusion
        pos_scores = pos_scores*tf.nn.sigmoid(self.pos_item_scores)*tf.nn.sigmoid(self.user_scores)
        neg_scores = neg_scores*tf.nn.sigmoid(self.neg_item_scores)*tf.nn.sigmoid(self.user_scores)

        # pos_scores = pos_scores*(tf.nn.sigmoid(self.pos_item_scores)+tf.nn.sigmoid(self.user_scores))
        # neg_scores = neg_scores*(tf.nn.sigmoid(self.neg_item_scores)+tf.nn.sigmoid(self.user_scores))


        self.mf_loss_ori = tf.reduce_mean(tf.negative(tf.log(tf.nn.sigmoid(pos_scores)+1e-10))+tf.negative(tf.log(1-tf.nn.sigmoid(neg_scores)+1e-10)))
        # second branch
        self.mf_loss_item = tf.reduce_mean(tf.negative(tf.log(tf.nn.sigmoid(self.pos_item_scores)+1e-10))+tf.negative(tf.log(1-tf.nn.sigmoid(self.neg_item_scores)+1e-10)))
        # third branch
        self.mf_loss_user = tf.reduce_mean(tf.negative(tf.log(tf.nn.sigmoid(self.user_scores)+1e-10))+tf.negative(tf.log(1-tf.nn.sigmoid(self.user_scores)+1e-10)))
        # unify
        mf_loss = self.mf_loss_ori + self.alpha*self.mf_loss_item + self.beta*self.mf_loss_user
        # regular
        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer/self.batch_size
        reg_loss = self.decay * regularizer
        return mf_loss, reg_loss
    
    def create_bce_loss_userc(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)   #users, pos_items, neg_items have the same shape
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        # item score
        # pos_items_stop = tf.stop_gradient(pos_items)
        # neg_items_stop = tf.stop_gradient(neg_items)
        pos_items_stop = pos_items
        neg_items_stop = neg_items
        self.pos_item_scores = tf.matmul(pos_items_stop,self.w)
        self.neg_item_scores = tf.matmul(neg_items_stop,self.w)
        self.rubi_ratings_userc = (self.batch_ratings-self.user_c)*tf.squeeze(tf.nn.sigmoid(self.pos_item_scores))
        self.direct_minus_ratings_userc = self.batch_ratings-self.user_c*tf.squeeze(tf.nn.sigmoid(self.pos_item_scores))
        # first branch
        # fusion
        pos_scores = (pos_scores-self.user_c)*tf.nn.sigmoid(self.pos_item_scores)
        neg_scores = (pos_scores-self.user_c)*tf.nn.sigmoid(self.neg_item_scores)
        self.mf_loss_ori = tf.reduce_mean(tf.negative(tf.log(tf.nn.sigmoid(pos_scores)+1e-10))+tf.negative(tf.log(1-tf.nn.sigmoid(neg_scores)+1e-10)))
        # second branch
        self.mf_loss_item = tf.reduce_mean(tf.negative(tf.log(tf.nn.sigmoid(self.pos_item_scores)+1e-10))+tf.negative(tf.log(1-tf.nn.sigmoid(self.neg_item_scores)+1e-10)))
        # unify
        mf_loss = self.mf_loss_ori #+ self.alpha*self.mf_loss_item
        # regular
        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer/self.batch_size
        reg_loss = self.decay * regularizer
        return mf_loss, reg_loss
        
        # self.rubi_ratings_userc = (self.batch_ratings-self.user_c)*tf.squeeze(tf.nn.sigmoid(self.pos_item_scores))
        # self.direct_minus_ratings_userc = self.batch_ratings-self.user_c*tf.squeeze(tf.nn.sigmoid(self.pos_item_scores))
        # pos_scores = (tf.reduce_sum(tf.multiply(users, pos_items), axis=1)-self.user_c)*tf.squeeze(tf.nn.sigmoid(self.pos_item_scores))
        # neg_scores = (tf.reduce_sum(tf.multiply(users, neg_items), axis=1)-self.user_c)*tf.squeeze(tf.nn.sigmoid(self.neg_item_scores))
        # # first branch
        # # fusion
        # mf_loss = tf.reduce_mean(tf.negative(tf.log(tf.nn.sigmoid(pos_scores)+1e-9))+tf.negative(tf.log(1-tf.nn.sigmoid(neg_scores)+1e-9)))
        # # regular
        # regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        # regularizer = regularizer/self.batch_size
        # reg_loss = self.decay * regularizer
        # return mf_loss, reg_loss

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)   #users, pos_items, neg_items have the same shape
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)

        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer/self.batch_size

        maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))

        mf_loss = tf.negative(tf.reduce_mean(maxi))
        reg_loss = self.decay * regularizer
        return mf_loss, reg_loss

    def create_bce_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)   #users, pos_items, neg_items have the same shape
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        # first branch
        # fusion
        mf_loss = tf.reduce_mean(tf.negative(tf.log(tf.nn.sigmoid(pos_scores)+1e-9))+tf.negative(tf.log(1-tf.nn.sigmoid(neg_scores)+1e-9)))
        # regular
        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer/self.batch_size
        reg_loss = self.decay * regularizer
        return mf_loss, reg_loss

    def create_bpr_loss2(self, users, const_embedding, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1) - tf.matmul(const_embedding, pos_items, transpose_a=False, transpose_b = True)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1) - tf.matmul(const_embedding, neg_items, transpose_a=False, transpose_b = True)

        regularizer = tf.nn.l2_loss(const_embedding)
        regularizer = regularizer/self.batch_size

        maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))

        mf_loss = tf.negative(tf.reduce_mean(maxi))
        reg_loss = self.decay * regularizer
        return mf_loss, reg_loss
    
    def create_bce_loss2(self, users, const_embedding, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1) - tf.matmul(const_embedding, pos_items, transpose_a=False, transpose_b = True)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1) - tf.matmul(const_embedding, neg_items, transpose_a=False, transpose_b = True)

        regularizer = tf.nn.l2_loss(const_embedding)
        regularizer = regularizer/self.batch_size

        mf_loss = tf.reduce_mean(tf.negative(tf.log(tf.nn.sigmoid(pos_scores)+1e-9))+tf.negative(tf.log(1-tf.nn.sigmoid(neg_scores)+1e-9)))
        reg_loss = self.decay * regularizer
        return mf_loss, reg_loss

    def update_c(self, sess, c):
        sess.run(tf.assign(self.rubi_c, c*tf.ones([1])))

    def _statistics_params(self):
        # number of params
        total_parameters = 0
        for variable in self.weights.values():
            shape = variable.get_shape()  # shape is an array of tf.Dimension
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        if self.verbose > 0:
            print("#params: %d" % total_parameters)
    
    def add_sess(self,sess):
        self.sess=sess
        self.model_type='o'
    
    def add_model_type(self,model_type):
        self.model_type=model_type
    
    def predict(self,users,items=None):
        if items==None:
            items = list(range(self.n_items))
        rate_batch = self.sess.run(self.batch_ratings, {self.users: users,
                                                                self.pos_items: items})
        if self.model_type == 'o':
            rate_batch = self.sess.run(self.batch_ratings, {self.users: users,
                                                            self.pos_items: items})
            #total_rate = np.vstack((total_rate, rate_batch))
        elif self.model_type == 'c':
            rate_batch = self.sess.run(self.user_const_ratings, {self.users: users,
                                                            self.pos_items: items})
        elif self.model_type == 'ic':
            rate_batch = self.sess.run(self.item_const_ratings, {self.users: users,
                                                            self.pos_items: items})
        elif self.model_type == 'rc':
            rate_batch = self.sess.run(self.user_rand_ratings, {self.users: users,
                                                            self.pos_items: items})
        elif self.model_type == 'irc':
            rate_batch = self.sess.run(self.item_rand_ratings, {self.users: users,
                                                            self.pos_items: items})
        elif self.model_type == 'rubi_c':
            rate_batch = self.sess.run(self.rubi_ratings, {self.users: users,
                                                            self.pos_items: items})
        elif self.model_type=="direct_minus_c":
            rate_batch = self.sess.run(self.direct_minus_ratings, {self.users: users,
                                                            self.pos_items: items})
        elif self.model_type == 'rubi_user_c':
            rate_batch = self.sess.run(self.rubi_ratings_userc, {self.users: users,
                                                            self.pos_items: items})
        elif self.model_type == 'rubi_both':
            rate_batch = self.sess.run(self.rubi_ratings_both, {self.users: users,
                                                            self.pos_items: items})
        else:
            print('model type error.')
            exit()
        return rate_batch


class DYNMF:
    def __init__(self, args, data_config,user_pop_num,item_pop_num):
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.decay = args.regs
        self.emb_dim = args.embed_size
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.verbose = args.verbose
        self.tau = args.tau
        self.temp = args.tau_info
        self.w_lambda = args.w_lambda
        if args.neg_sample>0:
            self.neg_sample=args.neg_sample
        else:
            self.neg_sample=args.batch_size-1
        self.pop_partition_user=user_pop_num
        self.pop_partition_item=item_pop_num

        #placeholders
        self.users = tf.placeholder(tf.int32, shape = (None,))
        self.pos_items = tf.placeholder(tf.int32, shape = (None,))
        self.neg_items = tf.placeholder(tf.int32, shape = (None,))

        self.users_pop = tf.placeholder(tf.int32, shape = (None,))
        self.pos_items_pop = tf.placeholder(tf.int32, shape = (None,))
        self.neg_items_pop = tf.placeholder(tf.int32, shape = (None,))
        self.item_mask = tf.placeholder(tf.int32, shape = (None,))

        #initiative weights
        self.weights = self.init_weights()

        #neting
        user_embedding = tf.nn.embedding_lookup(self.weights['user_embedding'], self.users)
        pos_item_embedding = tf.nn.embedding_lookup(self.weights['item_embedding'], self.pos_items)
        neg_item_embedding = tf.nn.embedding_lookup(self.weights['item_embedding'], self.neg_items)

        user_pop_embedding = tf.nn.embedding_lookup(self.weights['user_pop_embedding'], self.users_pop)
        pos_item_pop_embedding = tf.nn.embedding_lookup(self.weights['item_pop_embedding'], self.pos_items_pop)
        neg_item_pop_embedding = tf.nn.embedding_lookup(self.weights['item_pop_embedding'], self.neg_items_pop)

        self.batch_ratings = tf.matmul(user_embedding, pos_item_embedding, transpose_a=False, transpose_b = True)    #prediction, shape(user_embedding) != shape(pos_item_embedding)
        self.batch_ratings_pop = tf.matmul(user_pop_embedding, pos_item_pop_embedding, transpose_a=False, transpose_b = True)


        self.mf_loss1, self.mf_loss2, self.reg_loss, self.reg_loss_freeze, self.reg_loss_norm, self.mf_loss_ori= self.create_dyninfonce_loss(user_embedding, pos_item_embedding, neg_item_embedding, user_pop_embedding, pos_item_pop_embedding,neg_item_pop_embedding)

        self.loss = self.mf_loss1 + self.mf_loss2 + self.reg_loss

        self.freeze_loss = self.mf_loss2 + self.reg_loss_freeze

        self.mf_loss = self.mf_loss_ori + self.reg_loss_norm

        # self.opt = tf.train.RMSPropOptimizer(learning_rate = self.lr).minimize(self.loss)
        trainable_v1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'parameter')
        #print(trainable_v1)
        pop_list=[self.weights["user_pop_embedding"], self.weights["item_pop_embedding"]]
        ori_list=[self.weights['user_embedding'],self.weights['item_embedding']]
        #print(pop_list)
        #print(ori_list)
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss, var_list = trainable_v1)


        self.opt_freeze = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.freeze_loss, var_list = pop_list)
        self.opt_none_freeze = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss, var_list = ori_list)
        self.opt_mf = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.mf_loss, var_list = ori_list)
        #self.opt1 = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss, var_list = pop_list)
        #self.opt2 = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss, var_list = ori_list)


        self._statistics_params()



    def init_weights(self):
        weights = dict()
        self.initializer = tf.contrib.layers.xavier_initializer()
        initializer = self.initializer
        with tf.variable_scope('parameter'):
            weights["user_pop_embedding"] = tf.Variable(initializer([self.pop_partition_user, self.emb_dim]), name = 'user_pop_embedding')
            weights["item_pop_embedding"] = tf.Variable(initializer([self.pop_partition_item, self.emb_dim]), name = 'item_pop_embedding')
            weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name = 'user_embedding')
            weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name = 'item_embedding')

        
        with tf.variable_scope('const_embedding'):
            self.tau = tf.Variable(self.tau*tf.ones([1]), name = 'tau')
            self.w_lambda = tf.Variable(self.w_lambda*tf.ones([1]), name = 'tau')

        return weights

    
    
    
    
    
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

        # Scoring Type #1: sigmoid(dot prod)

        pos_item_score=tf.sigmoid(pos_item_prod)
        neg_item_score=tf.sigmoid(neg_item_prod)
        # pos_item_pop_score=tf.sigmoid(pos_item_pop_prod)/self.temp
        # neg_item_pop_score=tf.sigmoid(neg_item_pop_prod)/self.temp


        # Scoring Type #2: cosine similarity

        #pos_item_score=pos_item_prod/user_n2/pos_item_n2
        #neg_item_score=neg_item_prod/tiled_usr_n2/neg_item_n2
        pos_item_pop_score=pos_item_pop_prod/user_pop_n2/pos_item_pop_n2/self.temp
        neg_item_pop_score=neg_item_pop_prod/tiled_usr_pop_n2/neg_item_pop_n2/self.temp

        pos_item_score_mf_exp=tf.exp(pos_item_score/self.tau)
        neg_item_score_mf_exp=tf.reduce_sum(tf.exp(tf.reshape(neg_item_score/self.tau,[-1,self.neg_sample])),axis=1)


        loss_mf=tf.reduce_mean(tf.negative(tf.log(pos_item_score_mf_exp/(pos_item_score_mf_exp+neg_item_score_mf_exp))))



        neg_item_pop_score_exp=tf.reduce_sum(tf.exp(tf.reshape(neg_item_pop_score,[-1,self.neg_sample])),axis=1)
        pos_item_pop_score_exp=tf.exp(pos_item_pop_score)
        loss2=self.w_lambda*tf.reduce_mean(tf.negative(tf.log(pos_item_pop_score_exp/(pos_item_pop_score_exp+neg_item_pop_score_exp))))
        

        weighted_pos_item_score=tf.multiply(pos_item_score,tf.sigmoid(pos_item_pop_prod))/self.tau
        weighted_neg_item_score=tf.multiply(neg_item_score,tf.sigmoid(neg_item_pop_prod))/self.tau
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

        return loss1, loss2, reg_loss, reg_loss_freeze, reg_loss_norm, loss_mf

    def _statistics_params(self):
        # number of params
        total_parameters = 0
        for variable in self.weights.values():
            shape = variable.get_shape()  # shape is an array of tf.Dimension
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        if self.verbose > 0:
            print("#params: %d" % total_parameters)
    
    def add_sess(self,sess):
        self.sess=sess
    
    def update_lambda(self, sess, lamb):
        sess.run(tf.assign(self.w_lambda, lamb*tf.ones([1])))

    def update_tau(self, sess, tau):
        sess.run(tf.assign(self.tau, tau*tf.ones([1])))

    def predict(self,users,items=None):
        if items==None:
            items = list(range(self.n_items))
        rate_batch = self.sess.run(self.batch_ratings, {self.users: users,
                                                                self.pos_items: items})
        return rate_batch



class BIASMF:
    def __init__(self, args, data_config):
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.decay = args.regs
        self.emb_dim = args.embed_size
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.verbose = args.verbose
        self.c = args.c
        self.alpha = args.alpha
        #placeholders
        self.users = tf.placeholder(tf.int32, shape = (None,))
        self.pos_items = tf.placeholder(tf.int32, shape = (None,))
        self.neg_items = tf.placeholder(tf.int32, shape = (None,))

        #initiative weights
        self.weights = self.init_weights()

        #neting
        user_embedding = tf.nn.embedding_lookup(self.weights['user_embedding'], self.users)
        pos_item_embedding = tf.nn.embedding_lookup(self.weights['item_embedding'], self.pos_items)
        neg_item_embedding = tf.nn.embedding_lookup(self.weights['item_embedding'], self.neg_items)
        user_rand_embedding = tf.nn.embedding_lookup(self.weights['user_rand_embedding'], self.users)
        item_rand_embedding = tf.nn.embedding_lookup(self.weights['item_rand_embedding'], self.pos_items)
        self.const_embedding = self.weights['c']
        self.pos_item_bias = tf.nn.embedding_lookup(self.weights['item_bias'], self.pos_items)
        self.neg_item_bias = tf.nn.embedding_lookup(self.weights['item_bias'], self.neg_items)

        self.batch_ratings = tf.matmul(user_embedding, pos_item_embedding, transpose_a=False, transpose_b = True)    #prediction, shape(user_embedding) != shape(pos_item_embedding)
        self.user_const_ratings = self.batch_ratings - tf.matmul(self.const_embedding, pos_item_embedding, transpose_a=False, transpose_b = True)   #auto tile
        self.item_const_ratings = self.batch_ratings - tf.matmul(user_embedding, self.const_embedding, transpose_a=False, transpose_b = True)       #auto tile
        self.user_rand_ratings = self.batch_ratings - tf.matmul(user_rand_embedding, pos_item_embedding, transpose_a=False, transpose_b = True)
        self.item_rand_ratings = self.batch_ratings - tf.matmul(user_embedding, item_rand_embedding, transpose_a=False, transpose_b = True)


        self.mf_loss, self.reg_loss = self.create_bpr_loss(user_embedding, pos_item_embedding, neg_item_embedding)
        self.loss = self.mf_loss + self.reg_loss

        # self.opt = tf.train.RMSPropOptimizer(learning_rate = self.lr).minimize(self.loss)
        trainable_v1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'parameter')
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss, var_list = trainable_v1)
        # two branch
        self.w = tf.Variable(self.initializer([self.emb_dim,1]), name = 'item_branch')
        # two branch bpr
        self.mf_loss_two, self.reg_loss_two = self.create_bpr_loss_two_brach(user_embedding, pos_item_embedding, neg_item_embedding)
        self.loss_two = self.mf_loss_two + self.reg_loss_two
        self.opt_two = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_two)
        # two branch bce
        self.mf_loss_two_bce, self.reg_loss_two_bce = self.create_bce_loss_two_brach(user_embedding, pos_item_embedding, neg_item_embedding)
        self.loss_two_bce = self.mf_loss_two_bce + self.reg_loss_two_bce
        self.opt_two_bce = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_two_bce)
        # 2-stage training
        self.mf_loss2, self.reg_loss2 = self.create_bpr_loss2(user_embedding, self.const_embedding, pos_item_embedding, neg_item_embedding)
        self.loss2 = self.mf_loss2 + self.reg_loss2
        trainable_v2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'const_embedding')
        self.opt2 = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss2, var_list = trainable_v2)


        self.mf_loss2_bce, self.reg_loss2_bce = self.create_bce_loss2(user_embedding, self.const_embedding, pos_item_embedding, neg_item_embedding)
        self.loss2_bce = self.mf_loss2_bce + self.reg_loss2_bce
        self.opt2_bce = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss2_bce, var_list = trainable_v2)
        # self.opt = tf.train.RMSPropOptimizer(learning_rate = self.lr).minimize(self.loss)
        
        
        self.opt3 = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss2, var_list = [self.weights['user_embedding'],self.weights['item_embedding']])
        self.opt3_bce = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss2_bce, var_list = [self.weights['user_embedding'],self.weights['item_embedding']])

        self._statistics_params()

        self.mf_loss_bce, self.reg_loss_bce = self.create_bce_loss(user_embedding, pos_item_embedding, neg_item_embedding)
        self.loss_bce = self.mf_loss_bce + self.reg_loss_bce
        self.opt_bce = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_bce)



    def init_weights(self):
        weights = dict()
        self.initializer = tf.contrib.layers.xavier_initializer()
        initializer = self.initializer
        with tf.variable_scope('parameter'):
            weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name = 'user_embedding')
            weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name = 'item_embedding')
            weights['user_rand_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name = 'user_rand_embedding', trainable = False)
            weights['item_rand_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name = 'item_rand_embedding', trainable = False)
            weights['item_bias'] = tf.Variable(initializer([self.n_items]), name = 'item_bias')
        with tf.variable_scope('const_embedding'):
            self.rubi_c = tf.Variable(tf.zeros([1]), name = 'rubi_c')
            weights['c'] = tf.Variable(tf.zeros([1, self.emb_dim]), name = 'c')
        return weights

    def create_bpr_loss_two_brach(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1) + self.pos_item_bias   #users, pos_items, neg_items have the same shape
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1) + self.neg_item_bias
        # item stop


        # pos_items_stop = tf.stop_gradient(pos_items)
        # neg_items_stop = tf.stop_gradient(neg_items)
        pos_items_stop = pos_items
        neg_items_stop = neg_items

        self.pos_item_scores = tf.matmul(pos_items_stop,self.w)
        self.neg_item_scores = tf.matmul(neg_items_stop,self.w)
        # first branch
        pos_scores = pos_scores*tf.nn.sigmoid(self.pos_item_scores)
        neg_scores = neg_scores*tf.nn.sigmoid(self.neg_item_scores)
        maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))
        self.rubi_ratings = (self.batch_ratings-self.rubi_c)*tf.squeeze(tf.nn.sigmoid(self.pos_item_scores))
        self.direct_minus_ratings = self.batch_ratings-self.rubi_c*tf.squeeze(tf.nn.sigmoid(self.pos_item_scores))
        
        self.mf_loss_ori_bce = tf.negative(tf.reduce_mean(maxi))
        # second branch
        maxi_item = tf.log(tf.nn.sigmoid(self.pos_item_scores - self.neg_item_scores))
        self.mf_loss_item_bce = tf.negative(tf.reduce_mean(maxi_item))
        # unify
        mf_loss = self.mf_loss_ori_bce + self.alpha*self.mf_loss_item_bce
        # regular
        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer/self.batch_size

        
        reg_loss = self.decay * regularizer
        return mf_loss, reg_loss

    def create_bce_loss_two_brach(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1) + self.pos_item_bias   #users, pos_items, neg_items have the same shape
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1) + self.neg_item_bias
        # item score
        # pos_items_stop = tf.stop_gradient(pos_items)
        # neg_items_stop = tf.stop_gradient(neg_items)
        pos_items_stop = pos_items
        neg_items_stop = neg_items
        self.pos_item_scores = tf.matmul(pos_items_stop,self.w)
        self.neg_item_scores = tf.matmul(neg_items_stop,self.w)
        # self.rubi_ratings = (self.batch_ratings-self.rubi_c)*tf.squeeze(tf.nn.sigmoid(self.pos_item_scores))
        # self.direct_minus_ratings = self.batch_ratings-self.rubi_c*tf.squeeze(tf.nn.sigmoid(self.pos_item_scores))
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
        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer/self.batch_size
        reg_loss = self.decay * regularizer
        return mf_loss, reg_loss

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1) + self.pos_item_bias   #users, pos_items, neg_items have the same shape
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1) + self.neg_item_bias

        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer/self.batch_size

        maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))

        mf_loss = tf.negative(tf.reduce_mean(maxi))
        reg_loss = self.decay * regularizer
        return mf_loss, reg_loss

    def create_bce_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1) + self.pos_item_bias   #users, pos_items, neg_items have the same shape
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1) + self.neg_item_bias
        # first branch
        # fusion
        mf_loss = tf.reduce_mean(tf.negative(tf.log(tf.nn.sigmoid(pos_scores)+1e-9))+tf.negative(tf.log(1-tf.nn.sigmoid(neg_scores)+1e-9)))
        # regular
        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer/self.batch_size
        reg_loss = self.decay * regularizer
        return mf_loss, reg_loss

    def create_bpr_loss2(self, users, const_embedding, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1) - tf.matmul(const_embedding, pos_items, transpose_a=False, transpose_b = True) + self.pos_item_bias
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1) - tf.matmul(const_embedding, neg_items, transpose_a=False, transpose_b = True) + self.neg_item_bias

        regularizer = tf.nn.l2_loss(const_embedding)
        regularizer = regularizer/self.batch_size

        maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))

        mf_loss = tf.negative(tf.reduce_mean(maxi))
        reg_loss = self.decay * regularizer
        return mf_loss, reg_loss
    
    def create_bce_loss2(self, users, const_embedding, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1) - tf.matmul(const_embedding, pos_items, transpose_a=False, transpose_b = True) + self.pos_item_bias
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1) - tf.matmul(const_embedding, neg_items, transpose_a=False, transpose_b = True) + self.neg_item_bias

        regularizer = tf.nn.l2_loss(const_embedding)
        regularizer = regularizer/self.batch_size

        mf_loss = tf.reduce_mean(tf.negative(tf.log(tf.nn.sigmoid(pos_scores)+1e-9))+tf.negative(tf.log(1-tf.nn.sigmoid(neg_scores)+1e-9)))
        reg_loss = self.decay * regularizer
        return mf_loss, reg_loss

    def update_c(self, sess, c):
        sess.run(tf.assign(self.rubi_c, c*tf.ones([1])))

    def _statistics_params(self):
        # number of params
        total_parameters = 0
        for variable in self.weights.values():
            shape = variable.get_shape()  # shape is an array of tf.Dimension
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        if self.verbose > 0:
            print("#params: %d" % total_parameters)


class IPS_BPRMF:
    def __init__(self, args, data_config, p_matrix):
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.decay = args.regs
        self.emb_dim = args.embed_size
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.verbose = args.verbose
        self.c = args.c
        # self.p = p_matrix

        #placeholders
        self.users = tf.placeholder(tf.int32, shape = (None,))
        self.pos_items = tf.placeholder(tf.int32, shape = (None,))
        self.neg_items = tf.placeholder(tf.int32, shape = (None,))

        #initiative weights
        self.weights = self.init_weights()
        self.p = tf.constant(value = p_matrix, dtype = tf.float32)

        #neting

        user_embedding = tf.nn.embedding_lookup(self.weights['user_embedding'], self.users)
        pos_item_embedding = tf.nn.embedding_lookup(self.weights['item_embedding'], self.pos_items)
        neg_item_embedding = tf.nn.embedding_lookup(self.weights['item_embedding'], self.neg_items)
        self.pos_item_p = tf.nn.embedding_lookup(self.p, self.pos_items)
        self.neg_item_p = tf.nn.embedding_lookup(self.p, self.neg_items)
        user_rand_embedding = tf.nn.embedding_lookup(self.weights['user_rand_embedding'], self.users)
        item_rand_embedding = tf.nn.embedding_lookup(self.weights['item_rand_embedding'], self.pos_items)
        self.const_embedding = self.weights['const_embedding']

        self.batch_ratings = tf.matmul(user_embedding, pos_item_embedding, transpose_a=False, transpose_b = True)    #prediction, shape(user_embedding) != shape(pos_item_embedding)
        self.user_const_ratings = self.batch_ratings - tf.matmul(self.const_embedding, pos_item_embedding, transpose_a=False, transpose_b = True)   #auto tile
        self.item_const_ratings = self.batch_ratings - tf.matmul(user_embedding, self.const_embedding, transpose_a=False, transpose_b = True)       #auto tile
        self.user_rand_ratings = self.batch_ratings - tf.matmul(user_rand_embedding, pos_item_embedding, transpose_a=False, transpose_b = True)
        self.item_rand_ratings = self.batch_ratings - tf.matmul(user_embedding, item_rand_embedding, transpose_a=False, transpose_b = True)


        self.mf_loss, self.reg_loss = self.create_bpr_loss(user_embedding, pos_item_embedding, neg_item_embedding, self.pos_item_p, self.neg_item_p)
        self.loss = self.mf_loss + self.reg_loss

        # self.opt = tf.train.RMSPropOptimizer(learning_rate = self.lr).minimize(self.loss)
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
        self._statistics_params()



    def init_weights(self):
        weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()
        weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name = 'user_embedding')
        weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name = 'item_embedding')
        weights['user_rand_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name = 'user_rand_embedding', trainable = False)
        weights['item_rand_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name = 'item_rand_embedding', trainable = False)
        weights['const_embedding'] = tf.Variable(self.c*tf.ones([1, self.emb_dim]), name = 'const_embedding', trainable = False)
        return weights


    def create_bpr_loss(self, users, pos_items, neg_items, pos_item_p, neg_item_p):
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

        #self.temp1 = tf.negative(tf.log(pos_item_score_exp/(pos_item_score_exp+neg_item_score_exp)))
        mf_loss=tf.reduce_mean(tf.negative(tf.log(pos_item_score_exp/(pos_item_score_exp+neg_item_score_exp))))
        #maxi = tf.log(tf.nn.sigmoid(tf.divide(pos_scores, pos_item_p) - tf.divide(neg_scores, neg_item_p)))
        # tf.divide(pos_scores, pos_item_p) - tf.divide(neg_scores, neg_item_p)

        #mf_loss = tf.negative(tf.reduce_mean(maxi))
        reg_loss = self.decay * regularizer
        return mf_loss, reg_loss

    def _statistics_params(self):
        # number of params
        total_parameters = 0
        for variable in self.weights.values():
            shape = variable.get_shape()  # shape is an array of tf.Dimension
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        if self.verbose > 0:
            print("#params: %d" % total_parameters)

    def update_c(self, sess, c):
        sess.run(tf.assign(self.const_embedding, c*tf.ones([1, self.emb_dim])))

    
    def add_sess(self,sess):
        self.sess=sess

    def predict(self,users,items=None):
        if items==None:
            items = list(range(self.n_items))
        rate_batch = self.sess.run(self.batch_ratings, {self.users: users,
                                                                self.pos_items: items})
        return rate_batch

class CausalE:
    def __init__(self, args, data_config):
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.decay = args.regs
        self.emb_dim = args.embed_size
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.verbose = args.verbose
        self.cf_pen = args.cf_pen
        self.cf_loss = 0

        #placeholders
        self.users = tf.placeholder(tf.int32, shape = (None,))
        self.pos_items = tf.placeholder(tf.int32, shape = (None,))
        self.neg_items = tf.placeholder(tf.int32, shape = (None,))
        self.items = tf.placeholder(tf.int32, shape = (None,))
        self.reg_items = tf.placeholder(tf.int32, shape = (None,))

        #initiative weights
        self.weights = self.init_weights()

        #neting
        user_embedding = tf.nn.embedding_lookup(self.weights['user_embedding'], self.users)
        pos_item_embedding = tf.nn.embedding_lookup(self.weights['item_embedding'], self.pos_items)
        neg_item_embedding = tf.nn.embedding_lookup(self.weights['item_embedding'], self.neg_items)
        item_embedding = tf.nn.embedding_lookup(self.weights['item_embedding'], self.items)
        control_embedding = tf.stop_gradient(tf.nn.embedding_lookup(self.weights['item_embedding'], self.reg_items))
        self.batch_ratings = tf.matmul(user_embedding, pos_item_embedding, transpose_a=False, transpose_b = True)    #prediction, shape(user_embedding) != shape(pos_item_embedding)

        self.mf_loss, self.reg_loss, self.cf_loss = self.create_bpr_loss(user_embedding, pos_item_embedding, neg_item_embedding, item_embedding, control_embedding)
        self.loss = self.mf_loss + self.reg_loss + self.cf_loss

        # self.opt = tf.train.RMSPropOptimizer(learning_rate = self.lr).minimize(self.loss)
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
        self._statistics_params()



    def init_weights(self):
        weights = dict()
        self.initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None)#tf.contrib.layers.xavier_initializer()
        weights['user_embedding'] = tf.Variable(self.initializer([self.n_users, self.emb_dim]), name = 'user_embedding')
        weights['item_embedding'] = tf.Variable(self.initializer([self.n_items * 2, self.emb_dim]), name = 'item_embedding')
        
        return weights

    def create_bpr_loss(self, users, pos_items, neg_items, item_embed, control_embed):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)   #users, pos_items, neg_items have the same shape
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)

        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer/self.batch_size

        maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))

        mf_loss = tf.negative(tf.reduce_mean(maxi))
        reg_loss = self.decay * regularizer

        #counter factual loss

        #cf_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(tf.subtract(item_embed, control_embed)), axis=1))
        cf_loss = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(tf.nn.l2_normalize(item_embed,axis=0), tf.nn.l2_normalize(control_embed,axis=0)))))
        cf_loss = cf_loss * self.cf_pen #/ self.batch_size

        return mf_loss, reg_loss, cf_loss

    def _statistics_params(self):
        # number of params
        total_parameters = 0
        for variable in self.weights.values():
            shape = variable.get_shape()  # shape is an array of tf.Dimension
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        if self.verbose > 0:
            print("#params: %d" % total_parameters)

    def add_sess(self,sess):
        self.sess=sess

    def predict(self,users,items=None):
        if items==None:
            items = list(range(self.n_items))
        rate_batch = self.sess.run(self.batch_ratings, {self.users: users,
                                                                self.pos_items: items})
        return rate_batch
    
