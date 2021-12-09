import tensorflow as tf
import numpy as np
import os
import sys
import json
import random
import threading
import collections
import heapq
from tensorflow.python.client import device_lib
import math
import logging
from time import time
import multiprocessing
from tqdm import tqdm
from scipy.special import softmax, expit
from model import BPRMF, CausalE, IPS_BPRMF, BIASMF, DYNMF, DICE
from batch_test import *
from matplotlib import pyplot as plt
from copy import deepcopy
from evaluator import ProxyEvaluator
cores = multiprocessing.cpu_count() // 2

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'




def compute_2i_regularization_id(items, n_items):
    reg_ids = []
    for x in items:
        if x >= n_items:
            reg_ids.append(x - n_items)
            # reg_ids.append(x)
        elif x < n_items:
            reg_ids.append(x + n_items) # Add number of products to create the 2i representation 

    return reg_ids

def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)


def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, maxlen, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain
    """
    tp = 1. / np.log2(np.arange(2, k + 2))
    dcg_max = (tp[:min(maxlen, k)]).sum()
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def recall_at_k(r, k, all_pos_num):
    r = np.asfarray(r)[:k]
    return np.sum(r) / all_pos_num


def hit_at_k(r, k):
    r = np.array(r)[:k]
    if np.sum(r) > 0:
        return 1.
    else:
        return 0.

def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = dict()
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key = item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    
    return r

def get_performance(user_pos_test, r, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(precision_at_k(r, K))#P = TP/ (TP+FP)
        recall.append(recall_at_k(r, K, len(user_pos_test)))#R = TP/ (TP+FN)
        ndcg.append(ndcg_at_k(r, K, len(user_pos_test)))
        hit_ratio.append(hit_at_k(r, K))#HR = SIGMA(TP) / SIGMA(test_set)
    # print(hit_ratio)

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio)}

def test_one_user(x):
    # user u's ratings for user u
    rating = x[0]
    # uid
    u = x[1]
    # user u's items in the training set
    try:
        training_items = train_user_set[u]
    except Exception:
        training_items = []

    # user u's items in the test set
    user_pos_test = test_user_set[u]

    all_items = set(range(0, ITEM_NUM))
    test_items = list(all_items - set(training_items))
	#r为预测命中与否的集合，0未命中，1命中
    r = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)

    return get_performance(user_pos_test, r, Ks)

    
# def valid_one_user(x):
#     # user u's ratings for user u
#     rating = x[0]
#     #uid
#     u = x[1]
#     #user u's items in the training set
#     valid_items=x[2]
#     # try:
#     #     training_items = data.train_user_list[u]
#     # except Exception:
#     #     training_items = []
#     # #user u's items in the test set
#     user_pos_valid = data.valid_user_list[u]

#     # all_items = set(range(ITEM_NUM))
#     # valid_items = list(all_items - set(training_items))

# 	#r为预测命中与否的集合，0未命中，1命中
#     r = ranklist_by_sorted(user_pos_valid, valid_items, rating, Ks)

#     return get_performance(user_pos_valid, r, Ks)

def test(sess, dt, model, batch_test_flag = False, model_type = 'o', valid_set="test", item_pop_test=None, pop_exp = 0):

    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks))}
              

    global train_user_set, test_user_set


    train_user_set = deepcopy(dt.train_user_list)
    if valid_set=="test":
        test_user_set = deepcopy(dt.test_user_list)
    else:
        test_user_set = deepcopy(dt.valid_user_list)
    

    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE
    i_batch_size = BATCH_SIZE

    test_users = list(test_user_set.keys())
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0


    #total_rate = np.empty(shape=[0, ITEM_NUM])
    for u_batch_id in tqdm(range(n_user_batchs)):

        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]
        #print("start.")
        if batch_test_flag:

            n_item_batchs = ITEM_NUM // i_batch_size + 1
            rate_batch = np.zeros(shape=(len(user_batch), ITEM_NUM))
            i_count = 0
            for i_batch_id in tqdm(range(n_item_batchs)):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, ITEM_NUM)

                item_batch = range(i_start, i_end)
                if model_type == 'o':
                    i_rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
                                                                    model.pos_items: item_batch})
                elif model_type == 'c':
                    i_rate_batch = sess.run(model.user_const_ratings, {model.users: user_batch,
                                                                    model.pos_items: item_batch})
                elif model_type == 'ic':
                    i_rate_batch = sess.run(model.item_const_ratings, {model.users: user_batch,
                                                                    model.pos_items: item_batch})
                elif model_type == 'rc':
                    i_rate_batch = sess.run(model.user_rand_ratings, {model.users: user_batch,
                                                                    model.pos_items: item_batch})
                elif model_type == 'irc':
                    i_rate_batch = sess.run(model.item_rand_ratings, {model.users: user_batch,
                                                                    model.pos_items: item_batch})
                else:
                    print('model type error.')
                    exit()

                rate_batch[:, i_start: i_end] = i_rate_batch
                i_count += i_rate_batch.shape[1]

            assert i_count == ITEM_NUM

        else:
            item_batch = list(range(ITEM_NUM))
            if model_type == 'o':
                rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
                                                                model.pos_items: item_batch})
                #total_rate = np.vstack((total_rate, rate_batch))
            elif model_type == 'c':
                rate_batch = sess.run(model.user_const_ratings, {model.users: user_batch,
                                                                model.pos_items: item_batch})
            elif model_type == 'ic':
                rate_batch = sess.run(model.item_const_ratings, {model.users: user_batch,
                                                                model.pos_items: item_batch})
            elif model_type == 'rc':
                rate_batch = sess.run(model.user_rand_ratings, {model.users: user_batch,
                                                                model.pos_items: item_batch})
            elif model_type == 'irc':
                rate_batch = sess.run(model.item_rand_ratings, {model.users: user_batch,
                                                                model.pos_items: item_batch})
            elif model_type == 'rubi_c':
                rate_batch = sess.run(model.rubi_ratings, {model.users: user_batch,
                                                                model.pos_items: item_batch})
            elif model_type=="direct_minus_c":
                rate_batch = sess.run(model.direct_minus_ratings, {model.users: user_batch,
                                                                model.pos_items: item_batch})
            elif model_type == 'rubi_user_c':
                rate_batch = sess.run(model.rubi_ratings_userc, {model.users: user_batch,
                                                                model.pos_items: item_batch})
            elif model_type == 'rubi_both':
                rate_batch = sess.run(model.rubi_ratings_both, {model.users: user_batch,
                                                                model.pos_items: item_batch})
            elif model_type == "item_pop_test":
                rate_batch = sess.run(model.rubi_ratings_both_poptest, {model.users: user_batch,
                                                                model.pos_items: item_batch})
                # rate_batch = (rate_batch-np.min(rate_batch))*np.power(item_pop_test, pop_exp)
                rate_batch = np.ones_like(rate_batch)*np.power(item_pop_test, pop_exp)
            else:
                print('model type error.')
                exit()
        

        #print("end.")
        # item_acc_list = {}
        rate_batch = np.array(rate_batch)# (B, N)
        # for i in range(ITEM_NUM):
        #     item_acc_list[i] = 0
        all_items = set(range(ITEM_NUM))
        test_item_batch=[]
        # for j,rate_user in enumerate(rate_batch):
        #     user = user_batch[j]
        #     user_pos_test = data.test_user_list[user]
        #     train_items = data.train_user_list[user]
        #     test_items = list(all_items - set(train_items))
        #     test_item_batch.append(test_items)
            # item_score = dict()
            # for i in test_items:
            #     item_score[i] = rate_user[i]
            # K_max_item_score = heapq.nlargest(5, item_score, key = item_score.get)
            # for i in K_max_item_score:
            #     if i in user_pos_test:
            #         item_acc_list[i] += 1/len(data.test_item_list[i])

        user_batch_rating_uid = zip(rate_batch, user_batch)
        #batch_result=[]
        # for x in tqdm(user_batch_rating_uid):
        #     perf=test_one_user(x)
        #     batch_result.append(perf)

        batch_result = pool.map(test_one_user, user_batch_rating_uid)
        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision']/n_test_users
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users
            result['hit_ratio'] += re['hit_ratio']/n_test_users
    # print(result['hit_ratio'])
    # if model_type == 'o':
    #     # print('zk:', total_rate.shape, np.mean(total_rate))
    #     rate_sum, n = 0, 0
    #     # print(test_users)
    #     for user, items in train_user_set.items():
    #         if user not in test_users:
    #             continue
    #         idx = 0
    #         for u_id in test_users:
    #             if u_id == user:
    #                 break
    #             idx += 1
    #         rate_sum += np.sum(total_rate[idx,items])
    #         n += len(items)
        # print('pos rating:', rate_sum/n*1.0)


    assert count == n_test_users
    pool.close()
    return result

def early_stop(hr, ndcg, recall, precision, cur_epoch, config, stopping_step, flag_step = 1000):
    flag=0
    if hr >= config['best_hr']:
        stopping_step = 0
        flag=1
        config['best_hr'] = hr
        config['best_ndcg'] = ndcg
        config['best_recall'] = recall
        config['best_pre'] = precision
        config['best_epoch'] = cur_epoch
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger")
        should_stop = True
    else:
        should_stop = False

    return config, stopping_step, should_stop, flag


def ensureDir(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)


def merge_user_list(user_lists):
    out=collections.defaultdict(list)
    for user_list in user_lists:
        for key, item in user_list.items():
            out[key]=out[key]+item
    return out
cpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'CPU']
class sample_thread(threading.Thread):
    def __init__(self,pop=0):
        self.pop=pop
        threading.Thread.__init__(self)
    def run(self):
        if not self.pop:
            with tf.device(cpus[0]):
                self.data = data.sample()
        else:
            with tf.device(cpus[0]):
                self.data = data.sample_infonce(data.user_pop_idx,data.item_pop_idx)


if __name__ == '__main__':
    # random.seed(123)
    # tf.set_random_seed(123)
    #if args.valid_set=="test":
    pop_dict={}
    for user,items in data.train_user_list.items():
        for item in items:
            if item not in pop_dict:
                pop_dict[item]=0
            pop_dict[item]+=1
    
    sort_pop=sorted(pop_dict.items(), key=lambda item: item[1],reverse=True)
    pop_mask=[item[0] for item in sort_pop[:20]]
    print(pop_mask)


    if "new" in args.dataset:
        if args.model!="pop_test":
            eval_test_ood = ProxyEvaluator(data,data.train_user_list,data.test_user_list,top_k=Ks,dump_dict=merge_user_list([data.train_user_list,data.valid_user_list,data.test_id_user_list]))
            eval_test_id = ProxyEvaluator(data,data.train_user_list,data.test_id_user_list,top_k=Ks,dump_dict=merge_user_list([data.train_user_list,data.valid_user_list,data.test_user_list]))
            eval_valid = ProxyEvaluator(data,data.train_user_list,data.valid_user_list,top_k=Ks)
        else:
            eval_test_ood = ProxyEvaluator(data,data.train_user_list,data.test_user_list,top_k=Ks,dump_dict=merge_user_list([data.train_user_list,data.valid_user_list,data.test_id_user_list]),pop_mask=pop_mask)
            eval_test_id = ProxyEvaluator(data,data.train_user_list,data.test_id_user_list,top_k=Ks,dump_dict=merge_user_list([data.train_user_list,data.valid_user_list,data.test_user_list]),pop_mask=pop_mask)
            eval_valid = ProxyEvaluator(data,data.train_user_list,data.valid_user_list,top_k=Ks,pop_mask=pop_mask)
    else:
        eval_test_ood = ProxyEvaluator(data,data.train_user_list,data.test_user_list,top_k=Ks)
        eval_test_id = None
        eval_valid = eval_test_ood
    #if args.save_flag==1:
    weights_save_path='{}_{}_checkpoint/wd_{}_lr_{}_{}/'.format(args.model, args.dataset, args.wd, args.lr, args.saveID)
    ensureDir(weights_save_path)
    config = dict()
    config['n_users'] = data.n_users
    config['n_items'] = data.n_items
    model_type = ''
    if args.model == 'mf' or (args.model == 'CausalE' and args.skew == 2):
        model_type = 'mf'
        if args.train=="infonce":
            model = DYNMF(args,config,data.user_pop_num,data.item_pop_num)
        else:
            model = BPRMF(args, config)
        print('MF model.')
    elif args.model == 'CausalE':
        model_type = 'CausalE'
        model = CausalE(args, config)
        print('CausalE model.')
    elif args.model == 'IPSmf':
        model_type = 'IPSmf'
        p_matrix = dict()
        p = []
        for item, users in data.train_item_list.items():
            p_matrix[item] = (len(users)+1)/(data.n_users+1)
        for item in data.items:
            # print(item)
            if item not in p_matrix.keys():
                p_matrix[item] = 1/(data.n_users+1)
            p.append(p_matrix[item])
        print(p)
        model = IPS_BPRMF(args, config, p)
        print('IPS_MF model.')
    elif args.model == 'dice':
        model_type = 'IPSmf'
        p_matrix = dict()
        p = []
        for item, users in data.train_item_list.items():
            p_matrix[item] = (len(users)+1)/(data.n_users+1)
        for item in data.items:
            # print(item)
            if item not in p_matrix.keys():
                p_matrix[item] = 1/(data.n_users+1)
            p.append(p_matrix[item])
        model_type = 'dice'
        model = DICE(args,data)
        print("DICE model.")
    elif args.model == 'biasmf':
        model_type = 'mf'
        model = BIASMF(args, config)
        print('BIASMF model.')
    elif args.model == "dynmf" or args.model =="pop_test":
        model_type = 'dynmf'
        model=DYNMF(args,config,data.user_pop_num,data.item_pop_num)
        print('DYNMF model')

    vars_to_restore = []
    for var in tf.trainable_variables():
        if "item_embedding" in var.name:
            vars_to_restore.append(var)
    saver = tf.train.Saver(max_to_keep=5)
    # gpu_config = tf.ConfigProto()
    # gpu_config.gpu_options.allow_growth = True
    #sess = tf.Session(config = gpu_config)
    sess.run(tf.global_variables_initializer())

    
    model.add_sess(sess)

    if args.model=='pop_test':
        if "new" in args.dataset: 
            names=["valid","test_ood","test_id"]
            test_trials=[eval_valid,eval_test_ood,eval_test_id]
        else:
            names=["valid"]
            test_trials=[eval_valid]
        for w,eval in enumerate(test_trials):
            ret, _ = eval.evaluate(model)
            print(ret)

            n_ret={"recall":[ret[1],ret[1]],"hit_ratio":[ret[5],ret[5]],"precision":[ret[0],ret[0]],"ndcg":[ret[4],ret[4]]}
            #["Precision", "Recall", "MAP", "NDCG", "MRR", "HR"]
            ret=n_ret
            perf_str = 'pop_test, split=[%s], recall=[%.5f, %.5f], ' \
                'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                (names[w], ret['recall'][0], ret['recall'][-1],
                    ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                    ret['ndcg'][0], ret['ndcg'][-1])
            print(perf_str)
                # with open(weights_save_path + 'stats_{}.txt'.format(args.saveID),'a') as f:
                #     f.write(perf_str+"\n")
        



    #-----------CausalE----------
    elif model_type == 'CausalE':
        if args.pretrain == 0:
            t0 = time()
            loss_loger, pre_loger, rec_loger, ndcg_loger, auc_loger, hit_loger = [], [], [], [], [], []
            config["best_hr"], config["best_ndcg"], config['best_recall'], config['best_pre'], config["best_epoch"] = 0, 0, 0, 0, 0
            stopping_step = 0

            for epoch in range(args.epoch):
                t1 = time()
                loss, mf_loss, reg_loss, cf_loss = 0., 0., 0., 0.
                n_batch = data.n_train // args.batch_size + 1

                for idx in tqdm(range(n_batch)):
                    users, pos_items, neg_items = data.sample()#_cause()
                    items = pos_items + neg_items
                    reg_ids = compute_2i_regularization_id(items, ITEM_NUM)
                    _, batch_loss, batch_mf_loss, batch_reg_loss, batch_cf_loss = sess.run([model.opt, model.loss, model.mf_loss, model.reg_loss, model.cf_loss],
                                    feed_dict = {model.users: users,
                                                model.pos_items: pos_items,
                                                model.neg_items: neg_items,
                                                model.items: items,
                                                model.reg_items: reg_ids})
                    loss += batch_loss/n_batch
                    mf_loss += batch_mf_loss/n_batch
                    reg_loss += batch_reg_loss/n_batch
                    cf_loss += batch_cf_loss/n_batch

                if np.isnan(loss) == True:
                    print('ERROR: loss is nan.')
                    sys.exit()
                

                if args.verbose > 0 and epoch % args.verbose == 0:
                    perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f]' % (epoch, time()-t1, loss, mf_loss, reg_loss, cf_loss)
                    print(perf_str)
                    with open(weights_save_path + 'stats_{}.txt'.format(args.saveID),'a') as f:
                        f.write(perf_str+"\n")
                # print the test evaluation metrics each 10 epochs; pos:neg = 1:10.
                if (epoch + 1) % args.log_interval != 0:
                    continue

                t2 = time()
                #users_to_test = list(data.test_user_list.keys())
                if "new" in args.dataset: 
                    names=["valid","test_ood","test_id"]
                    test_trials=[eval_valid,eval_test_ood,eval_test_id]
                else:
                    names=["valid"]
                    test_trials=[eval_valid]
                for w,eval in enumerate(test_trials):
                    ret, _ = eval.evaluate(model)
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
                        perf_str = 'Epoch %d [%.1fs + %.1fs]: split=[%s], recall=[%.5f, %.5f], ' \
                            'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                            (epoch, t2 - t1, t3 - t2, names[w], ret['recall'][0], ret['recall'][-1],
                                ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                                ret['ndcg'][0], ret['ndcg'][-1])
                        print(perf_str)
                        with open(weights_save_path + 'stats_{}.txt'.format(args.saveID),'a') as f:
                            f.write(perf_str+"\n")

                # *********************************************************
                # save the user & item embeddings for pretraining.
                config, stopping_step, should_stop, best_flag = early_stop(ret['hit_ratio'][0], ret['ndcg'][0], ret['recall'][0], ret['precision'][0], epoch, config, stopping_step)
                if args.save_flag == 1:
                    if os.path.exists('{}_{}_checkpoint/wd_{}_lr_{}_{}/'.format(args.model, args.dataset, args.wd, args.lr, args.saveID)) == False:
                        os.makedirs('{}_{}_checkpoint/wd_{}_lr_{}_{}/'.format(args.model, args.dataset, args.wd, args.lr, args.saveID))
                    saver.save(sess, '{}_{}_checkpoint/wd_{}_lr_{}_{}/{}_ckpt.ckpt'.format(args.model, args.dataset, args.wd, args.lr, args.saveID, epoch))
                    if best_flag:
                        config['best_name']='{}_{}_checkpoint/wd_{}_lr_{}_{}/{}_ckpt.ckpt'.format(args.model, args.dataset, args.wd, args.lr, args.saveID, epoch)

                if should_stop:
                    print("{} dataset best epoch{}: hr:{} ndcg:{} recall:{} precision:{}".format(args.dataset, config['best_epoch'],config['best_hr'],config['best_ndcg'], config['best_recall'], config['best_pre']))
                    logging.info("{} dataset best epoch{}: hr:{} ndcg:{} recall:{} precision:{}".format(args.dataset, config['best_epoch'],config['best_hr'],config['best_ndcg'], config['best_recall'], config['best_pre']))
                    break
            

            print('#loading best models at epoch {}'.format(config['best_epoch']))
            saver.restore(sess, config['best_name'])
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
                    
        else:
            pass
            # print('#load existing models.')
            # model_file = tf.train.latest_checkpoint('{}_{}_checkpoint/wd_{}_lr_{}/'.format(args.model, args.dataset, args.wd, args.lr))
            # saver.restore(sess, model_file)
    elif model_type=="dice":
        if args.pretrain==0:
            t0 = time()
            loss_loger, pre_loger, rec_loger, ndcg_loger, auc_loger, hit_loger = [], [], [], [], [], []
            config["best_hr"], config["best_ndcg"], config['best_recall'], config['best_pre'], config["best_epoch"] = 0, 0, 0, 0, 0
            #config['best_tau_hr'], config['best_tau_epoch'], config['best_tau'] = 0, 0, 0.0
            stopping_step = 0
            cur_tau=args.tau
            cur_w_lambda=args.w_lambda

            perf=model.eval(args)
            ret, _ = eval_valid.evaluate(model)
            print(ret)
            #ret, _ = eval_valid.evaluate(model)
            
            t3 = time()
            #print(ret)
            

            for epoch in range(args.epoch):
                t1 = time()
                perf=model.train_1_epoch(args)
                # print the test evaluation metrics each 10 epochs; pos:neg = 1:10.
                if (epoch + 1) % args.log_interval != 0:
                    if args.verbose > 0 and epoch % args.verbose == 0:
                        perf_str = 'Epoch %d [%.1fs]: ' % (epoch, time()-t1) + perf
                        print(perf_str)
                        with open(weights_save_path + 'stats_{}.txt'.format(args.saveID),'a') as f:
                            f.write(perf_str+"\n")
                    continue
                t2 = time()
                # start testing
                # if args.valid_set=="test":
                #for tau in np.linspace(args.start, args.end, args.step):
                #model.update_tau(sess, tau)

                
                
                if "new" in args.dataset: 
                    names=["valid","test_ood","test_id"]
                    test_trials=[eval_valid,eval_test_ood,eval_test_id]
                else:
                    names=["valid"]
                    test_trials=[eval_valid]
                for w,eval in enumerate(test_trials):
                    ret, _ = eval.evaluate(model)
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
                        perf_str = 'Epoch %d [%.1fs + %.1fs]: %s split=[%s], recall=[%.5f, %.5f], ' \
                            'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                            (epoch, t2 - t1, t3 - t2, perf, names[w], ret['recall'][0], ret['recall'][-1],
                                ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                                ret['ndcg'][0], ret['ndcg'][-1])
                        print(perf_str)
                        with open(weights_save_path + 'stats_{}.txt'.format(args.saveID),'a') as f:
                            f.write(perf_str+"\n")
                model.eval(args)
                        
                # *********************************************************
                # save the user & item embeddings for pretraining.
                config, stopping_step, should_stop, best_flag = early_stop(ret['hit_ratio'][0], ret['ndcg'][0], ret['recall'][0], ret['precision'][0], epoch, config, stopping_step)
                if args.save_flag == 1:
                    if os.path.exists('{}_{}_checkpoint/wd_{}_lr_{}_{}/'.format(args.model, args.dataset, args.wd, args.lr, args.saveID)) == False:
                        os.makedirs('{}_{}_checkpoint/wd_{}_lr_{}_{}/'.format(args.model, args.dataset, args.wd, args.lr, args.saveID))
                    saver.save(sess, '{}_{}_checkpoint/wd_{}_lr_{}_{}/{}_ckpt.ckpt'.format(args.model, args.dataset, args.wd, args.lr, args.saveID, epoch))
                    if best_flag:
                        config['best_name']='{}_{}_checkpoint/wd_{}_lr_{}_{}/{}_ckpt.ckpt'.format(args.model, args.dataset, args.wd, args.lr, args.saveID, epoch)
                if should_stop and args.early_stop == 1:
                    print("{} dataset best epoch{}: hr:{} ndcg:{} recall:{} precision:{}".format(args.dataset, config['best_epoch'],config['best_hr'],config['best_ndcg'], config['best_recall'], config['best_pre']))
                    logging.info("{} dataset best epoch{}: hr:{} ndcg:{} recall:{} precision:{}".format(args.dataset, config['best_epoch'],config['best_hr'],config['best_ndcg'], config['best_recall'], config['best_pre']))

                    with open('{}_{}_checkpoint/wd_{}_lr_{}_{}/best_epoch.txt'.format(args.model, args.dataset, args.wd, args.lr, args.saveID),'w') as f:
                        print(config['best_epoch'], file = f)
                    break
            
            # test ood/id
            saver.restore(sess, config['best_name'])
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
        else:
            print('#load existing models.')

            
            overlap_num=[10,20,30,40,50]
            poptest=[{},{},{},{},{}]
            p_all=np.array(p)
            rank=np.argsort(-p_all)
            for user in range(data.n_users):
                dump=[]
                if user in data.train_user_list:
                    dump+=data.train_user_list[user]
                if user in data.valid_user_list:
                    dump+=data.valid_user_list[user]

                ptr=0
                num=0
                my_rank=[]
                while num<50:
                    if rank[ptr] not in dump:
                        my_rank.append(rank[ptr])
                        num+=1
                    ptr+=1


                
                for i,num in enumerate(overlap_num):
                    poptest[i][user]=my_rank[:num]
            
            model_file = '/storage/wcma/MACR/dice_tencent.new_checkpoint/wd_1e-05_lr_0.0001_3/399_ckpt.ckpt'
            saver.restore(sess, model_file)

            ret, _ = eval_test_ood.evaluate(model)
            n_ret = {"recall":ret[1], "hit_ratio":ret[5], "precision":ret[0], "ndcg":ret[4]}
            ret=n_ret
            perf_str = 'test_ood: recall={}, ' \
                                'precision={}, hit={}, ndcg={}'.format(str(ret["recall"]),
                                    str(ret['precision']), str(ret['hit_ratio']), str(ret['ndcg']))

            print(perf_str)
            # with open(weights_save_path + 'stats_{}.txt'.format(args.saveID),'a') as f:
            #     f.write(perf_str+"\n")
            
            # if "new" in args.dataset:
            ret, _ = eval_test_id.evaluate(model)
            n_ret = {"recall":ret[1], "hit_ratio":ret[5], "precision":ret[0], "ndcg":ret[4]}
            ret=n_ret
            perf_str = 'test_id: recall={}, ' \
                                'precision={}, hit={}, ndcg={}'.format(str(ret["recall"]),
                                    str(ret['precision']), str(ret['hit_ratio']), str(ret['ndcg']))
            print(perf_str)
    
                
            
            for i,num in enumerate(overlap_num):
                pop_eval=ProxyEvaluator(data,data.train_user_list,poptest[i],top_k=[num],dump_dict=merge_user_list([data.train_user_list,data.valid_user_list]))
                ret, _ = pop_eval.evaluate(model)
                print("Overlap@",num," ",ret[1])
            
            # perf=model.eval(args)
            # print(perf)

            # #print('#loading best models at epoch {}'.format(config['best_epoch'])
            
            #     print(perf_str)
            #     with open(weights_save_path + 'stats_{}.txt'.format(args.saveID),'a') as f:
            #         f.write(perf_str+"\n")


            #print(perf_str)
    



    #Our dynmf model
    elif model_type=="dynmf":
        if args.pretrain == 0:
            t0 = time()
            loss_loger, pre_loger, rec_loger, ndcg_loger, auc_loger, hit_loger = [], [], [], [], [], []
            config["best_hr"], config["best_ndcg"], config['best_recall'], config['best_pre'], config["best_epoch"] = 0, 0, 0, 0, 0
            #config['best_tau_hr'], config['best_tau_epoch'], config['best_tau'] = 0, 0, 0.0
            stopping_step = 0
            cur_tau=args.tau
            cur_w_lambda=args.w_lambda

            for epoch in range(args.epoch):
                t1 = time()
                loss, mf_loss1, mf_loss2, reg_loss  = 0., 0., 0., 0.
                n_batch = data.n_train // args.batch_size + 1
                if epoch >= args.warm_up:
                    if cur_tau > args.tau_cut:
                        cur_tau=cur_tau*args.tau_decay
                        model.update_tau(sess, cur_tau)
                    if cur_w_lambda > args.lambda_cut:
                        cur_w_lambda=cur_w_lambda*args.w_lambda_decay
                        model.update_lambda(sess,cur_w_lambda)
                

                cur_opt=model.opt
                if args.freeze==1:
                    if epoch<args.freeze_epoch:
                        cur_opt=model.opt_freeze
                    else:
                        cur_opt=model.opt_none_freeze

                
                sample_last = sample_thread(pop=1)
                sample_last.start()
                sample_last.join()


                for idx in tqdm(range(n_batch)):
                    users, pos_items, neg_items, users_pop, pos_items_pop, neg_items_pop = sample_last.data#data.sample_infonce(data.user_pop_idx,data.item_pop_idx)
                    sample_next = sample_thread(pop=1)
                    sample_next.start()
                    sample_next.join()
                    _, batch_loss, batch_mf_loss1, batch_mf_loss2, batch_reg_loss = sess.run([cur_opt, model.loss, model.mf_loss1, model.mf_loss2,model.reg_loss],
                                    feed_dict = {model.users: users,
                                                model.pos_items: pos_items,
                                                model.neg_items: neg_items,
                                                model.users_pop: users_pop,
                                                model.pos_items_pop: pos_items_pop,
                                                model.neg_items_pop: neg_items_pop})
                    sample_last=sample_next
                    mf_loss1 += batch_mf_loss1/n_batch
                    mf_loss2 += batch_mf_loss2/n_batch
                    reg_loss += batch_reg_loss/n_batch
                    loss += batch_loss/n_batch
                if np.isnan(loss) == True:
                    print('ERROR: loss is nan.')
                    sys.exit()

                # print the test evaluation metrics each 10 epochs; pos:neg = 1:10.
                if (epoch + 1) % args.log_interval != 0 or epoch < args.freeze_epoch:
                    if args.verbose > 0 and epoch % args.verbose == 0:
                        perf_str = 'Epoch %d [%.1fs]: tau, lambda==[%.2f, %.2f], train==[%.5f=%.5f + %.5f + %.5f]' % (epoch, time()-t1, cur_tau, cur_w_lambda, loss, mf_loss1, mf_loss2, reg_loss)
                        print(perf_str)
                        with open(weights_save_path + 'stats_{}.txt'.format(args.saveID),'a') as f:
                            f.write(perf_str+"\n")
                    continue
                t2 = time()
                # start testing
                # if args.valid_set=="test":
                #for tau in np.linspace(args.start, args.end, args.step):
                #model.update_tau(sess, tau)
                
                if "new" in args.dataset: 
                    names=["valid","test_ood","test_id"]
                    test_trials=[eval_valid,eval_test_ood,eval_test_id]
                else:
                    names=["valid"]
                    test_trials=[eval_valid]
                for w,eval in enumerate(test_trials):
                    ret, _ = eval.evaluate(model)
                    t3 = time()
                    print(ret)

                    n_ret={"recall":[ret[1],ret[1]],"hit_ratio":[ret[5],ret[5]],"precision":[ret[0],ret[0]],"ndcg":[ret[4],ret[4]]}
                    #["Precision", "Recall", "MAP", "NDCG", "MRR", "HR"]
                    ret=n_ret
                    if w==0:
                        loss_loger.append(loss)
                        rec_loger.append(ret['recall'][0])
                        pre_loger.append(ret['precision'][0])
                        ndcg_loger.append(ret['ndcg'][0])
                        hit_loger.append(ret['hit_ratio'][0])


                        best_hr = ret['hit_ratio'][0]
                        best_recall=ret['recall'][0]
                        best_pre=ret['precision'][0]
                        best_ndcg=ret['ndcg'][0]

                    if args.verbose > 0:
                        perf_str = 'Epoch %d [%.1fs + %.1fs]: tau, lambda==[%.2f, %.2f], train==[%.8f=%.8f + %.8f + %.8f], split=[%s], recall=[%.5f, %.5f], ' \
                            'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                            (epoch, t2 - t1, t3 - t2, cur_tau, cur_w_lambda, loss, mf_loss1, mf_loss2, reg_loss, names[w], ret['recall'][0], ret['recall'][-1],
                                ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                                ret['ndcg'][0], ret['ndcg'][-1])
                        print(perf_str)
                        with open(weights_save_path + 'stats_{}.txt'.format(args.saveID),'a') as f:
                            f.write(perf_str+"\n")
                        
                # *********************************************************
                # save the user & item embeddings for pretraining.
                config, stopping_step, should_stop, best_flag = early_stop(ret['hit_ratio'][0], ret['ndcg'][0], ret['recall'][0], ret['precision'][0], epoch, config, stopping_step)
                if args.save_flag == 1:
                    if os.path.exists('{}_{}_checkpoint/wd_{}_lr_{}_{}/'.format(args.model, args.dataset, args.wd, args.lr, args.saveID)) == False:
                        os.makedirs('{}_{}_checkpoint/wd_{}_lr_{}_{}/'.format(args.model, args.dataset, args.wd, args.lr, args.saveID))
                    saver.save(sess, '{}_{}_checkpoint/wd_{}_lr_{}_{}/{}_ckpt.ckpt'.format(args.model, args.dataset, args.wd, args.lr, args.saveID, epoch))
                    if best_flag:
                        config['best_name']='{}_{}_checkpoint/wd_{}_lr_{}_{}/{}_ckpt.ckpt'.format(args.model, args.dataset, args.wd, args.lr, args.saveID, epoch)
                if should_stop and args.early_stop == 1:
                    print("{} dataset best epoch{}: hr:{} ndcg:{} recall:{} precision:{}".format(args.dataset, config['best_epoch'],config['best_hr'],config['best_ndcg'], config['best_recall'], config['best_pre']))
                    logging.info("{} dataset best epoch{}: hr:{} ndcg:{} recall:{} precision:{}".format(args.dataset, config['best_epoch'],config['best_hr'],config['best_ndcg'], config['best_recall'], config['best_pre']))

                    with open('{}_{}_checkpoint/wd_{}_lr_{}_{}/best_epoch.txt'.format(args.model, args.dataset, args.wd, args.lr, args.saveID),'w') as f:
                        print(config['best_epoch'], file = f)
                    break
            
            # test ood/id
            print('#loading best models at epoch {}'.format(config['best_epoch']))
            saver.restore(sess, config['best_name'])
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

        

            
        # pretrain
        else:
            print('#load existing models.')

            model_file = 'mf_addressa_checkpoint/wd_1e-05_lr_0.001_mfnormalbce/669_ckpt.ckpt'
            saver.restore(sess, model_file)
            ret = test(sess, data, model, valid_set="test")
            perf_str = ' recall={}, ' \
                                    'precision={}, hit={}, ndcg={}'.format(str(ret["recall"]),
                                        str(ret['precision']), str(ret['hit_ratio']), str(ret['ndcg']))
            print(perf_str)
            exit()

    # MF
    else:
        # no pretrain
        if args.pretrain == 0:
            t0 = time()
            loss_loger, pre_loger, rec_loger, ndcg_loger, auc_loger, hit_loger = [], [], [], [], [], []
            config["best_hr"], config["best_ndcg"], config['best_recall'], config['best_pre'], config["best_epoch"] = 0, 0, 0, 0, 0
            config['best_c_hr'], config['best_c_epoch'], config['best_c'] = 0, 0, 0.0
            stopping_step = 0
            perf_dict={"Precision":[],"Recall":[],"MAP":[],"NDCG":[],"MRR":[],"HR":[]}
            for epoch in range(args.epoch):
                t1 = time()
                loss, mf_loss, reg_loss = 0., 0., 0.
                n_batch = data.n_train // args.batch_size + 1
                

                for idx in tqdm(range(n_batch)):
                    if args.train!="infonce":
                        users, pos_items, neg_items = data.sample()
                    else:
                        users, pos_items, neg_items, users_pop, pos_items_pop, neg_items_pop = data.sample_infonce(data.user_pop_idx,data.item_pop_idx)

                    if args.train=="normal":
                        _, batch_loss, batch_mf_loss, batch_reg_loss = sess.run([model.opt_mf,model.mf_loss, model.mf_loss_ori, model.reg_loss_norm],
                                        feed_dict = {model.users: users,
                                                    model.pos_items: pos_items,
                                                    model.neg_items: neg_items})
                        #print(debug)
                        #input()
                    elif args.train=="infonce":
                        _, batch_loss, batch_mf_loss, batch_reg_loss = sess.run([model.opt,model.loss, model.mf_loss, model.reg_loss],
                                        feed_dict = {model.users: users,
                                                model.pos_items: pos_items,
                                                model.neg_items: neg_items,
                                                model.users_pop: users_pop,
                                                model.pos_items_pop: pos_items_pop,
                                                model.neg_items_pop: neg_items_pop})

                    elif args.train=="rubi":
                        _, batch_loss, batch_mf_loss, batch_reg_loss = sess.run([model.opt_two, model.loss_two, model.mf_loss_two, model.reg_loss_two],
                                        feed_dict = {model.users: users,
                                                    model.pos_items: pos_items,
                                                    model.neg_items: neg_items})
                    elif args.train=="rubibce":
                        _, batch_loss, batch_mf_loss, batch_reg_loss = sess.run([model.opt_two_bce, model.loss_two_bce, model.mf_loss_two_bce, model.reg_loss_two_bce],
                                        feed_dict = {model.users: users,
                                                    model.pos_items: pos_items,
                                                    model.neg_items: neg_items})   
                    elif args.train=="normalbce":
                        _, batch_loss, batch_mf_loss, batch_reg_loss = sess.run([model.opt_bce, model.loss_bce, model.mf_loss_bce, model.reg_loss_bce],
                                        feed_dict = {model.users: users,
                                                    model.pos_items: pos_items,
                                                    model.neg_items: neg_items})
                    elif args.train=="rubibceboth":
                        _, batch_loss, batch_mf_loss, batch_reg_loss = sess.run([model.opt_two_bce_both, model.loss_two_bce_both, model.mf_loss_two_bce_both, model.reg_loss_two_bce_both],
                                        feed_dict = {model.users: users,
                                                    model.pos_items: pos_items,
                                                    model.neg_items: neg_items}) 
                        # print(batch_mf_loss, batch_reg_loss)
                    # _, batch_loss, batch_mf_loss, batch_reg_loss = sess.run([model.opt_two_bce, model.loss_two_bce, model.mf_loss_ori, model.mf_loss_item],
                    #             feed_dict = {model.users: users,
                    #                         model.pos_items: pos_items,
                    #                         model.neg_items: neg_items})  
                    # _, batch_loss, batch_mf_loss, batch_reg_loss = sess.run([model.opt_two, model.loss_two, model.mf_loss_ori_bce, model.mf_loss_item_bce],
                    #                 feed_dict = {model.users: users,
                    #                             model.pos_items: pos_items,
                    #                             model.neg_items: neg_items})
                    # _, batch_loss, batch_mf_loss, batch_reg_loss = sess.run([model.opt_two_bce, model.loss_two_bce, model.mf_loss_two_bce, model.reg_loss_two_bce],
                    #                 feed_dict = {model.users: users,
                    #                             model.pos_items: pos_items,
                    #                             model.neg_items: neg_items})      
                    loss += batch_loss/n_batch
                    mf_loss += batch_mf_loss/n_batch
                    reg_loss += batch_reg_loss/n_batch
                if np.isnan(loss) == True:
                    print('ERROR: loss is nan.')
                    sys.exit()

                # print the test evaluation metrics each 10 epochs; pos:neg = 1:10.
                if (epoch + 1) % args.log_interval != 0:
                    if args.verbose > 0 and epoch % args.verbose == 0:
                        perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (epoch, time()-t1, loss, mf_loss, reg_loss)
                        print(perf_str)
                        with open(weights_save_path + 'stats_{}.txt'.format(args.saveID),'a') as f:
                            f.write(perf_str+"\n")
                    continue

                t2 = time()
                # start testing
                if args.test=="normal" or args.test == 'rubi_user_wise':
                    ret, _ = eval_valid.evaluate(model)
                    # ret = test(sess, model, users_to_test)
                    t3 = time()
                    print(ret)

                    for i,key in enumerate(perf_dict.keys()):
                        perf_dict[key].append(str(ret[i]))
                    

                    n_ret={"recall":[ret[1],ret[1]],"hit_ratio":[ret[5],ret[5]],"precision":[ret[0],ret[0]],"ndcg":[ret[4],ret[4]]}
                    ret=n_ret
                    loss_loger.append(loss)
                    rec_loger.append(ret['recall'][0])
                    pre_loger.append(ret['precision'][0])
                    ndcg_loger.append(ret['ndcg'][0])
                    hit_loger.append(ret['hit_ratio'][0])
                    if args.verbose > 0:
                        perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.8f=%.8f + %.8f], recall=[%.5f, %.5f], ' \
                                'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                                (epoch, t2 - t1, t3 - t2, loss, mf_loss, reg_loss, ret['recall'][0], ret['recall'][-1],
                                    ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                                    ret['ndcg'][0], ret['ndcg'][-1])
                        print(perf_str)
                        with open(weights_save_path + 'stats_{}.txt'.format(args.saveID),'a') as f:
                            f.write(perf_str+"\n")
                        with open(weights_save_path + 'perf_{}.json'.format(args.saveID),'w') as f:
                            json.dump(perf_dict,f)
                elif args.test=="rubi":
                    print('Epoch %d'%(epoch))
                    with open(weights_save_path + 'stats_{}.txt'.format(args.saveID),'a') as f:
                        f.write('Epoch %d'%(epoch)+"\n")
                    best_c = 0
                    best_hr = 0
                    best_recall=0
                    best_ndcg=0
                    best_pre=0
                    best_flag=0
                    for c in np.linspace(args.start, args.end, args.step):
                        model.update_c(sess, c)
                        if args.train == 'rubibceboth':
                            model.add_model_type("rubi_both")
                            ret, _ = eval_valid.evaluate(model)

                        #ret = test(sess, data, model, model_type="rubi_both", valid_set="test")
                        else:
                            model.add_model_type("rubi_c")
                            ret, _ = eval_valid.evaluate(model)
                        t3 = time()
                        print(ret)

                        n_ret={"recall":[ret[1],ret[1]],"hit_ratio":[ret[5],ret[5]],"precision":[ret[0],ret[0]],"ndcg":[ret[4],ret[4]]}
                        #["Precision", "Recall", "MAP", "NDCG", "MRR", "HR"]
                        ret=n_ret
                        
                        loss_loger.append(loss)
                        rec_loger.append(ret['recall'][0])
                        pre_loger.append(ret['precision'][0])
                        ndcg_loger.append(ret['ndcg'][0])
                        hit_loger.append(ret['hit_ratio'][0])

                        if ret['hit_ratio'][0] > best_hr:
                            best_hr = ret['hit_ratio'][0]
                            best_recall=ret['recall'][0]
                            best_pre=ret['precision'][0]
                            best_ndcg=ret['ndcg'][0]
                            best_c = c

                        if args.verbose > 0:
                            perf_str = 'c:%.2f [%.1fs + %.1fs]: train==[%.8f=%.8f + %.8f], recall=[%.5f, %.5f], ' \
                                    'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                                    (c, t2 - t1, t3 - t2, loss, mf_loss, reg_loss, ret['recall'][0], ret['recall'][-1],
                                        ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                                        ret['ndcg'][0], ret['ndcg'][-1])
                            print(perf_str)
                            with open(weights_save_path + 'stats_{}.txt'.format(args.saveID),'a') as f:
                                f.write(perf_str+"\n")

                    if best_hr > config['best_c_hr']:
                        best_flag=1
                        config['best_c_hr'] = best_hr
                        config['best_c_ndcg'] = best_ndcg
                        config['best_c_precision'] = best_pre
                        config['best_c_recall'] = best_recall
                        config['best_c_epoch'] = epoch
                        config['best_c'] = best_c
                    
                    ret['hit_ratio'][0]=best_hr
                    ret['recall'][0]=best_recall
                    ret['precision'][0]=best_pre
                    ret['ndcg'][0]=best_ndcg
                    


                # *********************************************************
                # save the user & item embeddings for pretraining.
                if args.test=="rubi":
                    model.update_c(sess,best_c)
                config, stopping_step, should_stop,best_flag = early_stop(ret['hit_ratio'][0], ret['ndcg'][0], ret['recall'][0], ret['precision'][0], epoch, config, stopping_step)
                if args.save_flag == 1:
                    if os.path.exists('{}_{}_checkpoint/wd_{}_lr_{}_{}/'.format(args.model, args.dataset, args.wd, args.lr, args.saveID)) == False:
                        os.makedirs('{}_{}_checkpoint/wd_{}_lr_{}_{}/'.format(args.model, args.dataset, args.wd, args.lr, args.saveID))
                    saver.save(sess, '{}_{}_checkpoint/wd_{}_lr_{}_{}/{}_ckpt.ckpt'.format(args.model, args.dataset, args.wd, args.lr, args.saveID, epoch))
                    if best_flag:
                        config['best_name']='{}_{}_checkpoint/wd_{}_lr_{}_{}/{}_ckpt.ckpt'.format(args.model, args.dataset, args.wd, args.lr, args.saveID, epoch)
                if should_stop and args.early_stop == 1:
                    print("{} dataset best epoch{}: hr:{} ndcg:{} recall:{} precision:{}".format(args.dataset, config['best_epoch'],config['best_hr'],config['best_ndcg'], config['best_recall'], config['best_pre']))
                    logging.info("{} dataset best epoch{}: hr:{} ndcg:{} recall:{} precision:{}".format(args.dataset, config['best_epoch'],config['best_hr'],config['best_ndcg'], config['best_recall'], config['best_pre']))

                    with open('{}_{}_checkpoint/wd_{}_lr_{}_{}/best_epoch.txt'.format(args.model, args.dataset, args.wd, args.lr, args.saveID),'w') as f:
                        print(config['best_epoch'], file = f)

                    if args.test=='rubi':
                        with open('{}_{}_checkpoint/wd_{}_lr_{}_{}/best_c.txt'.format(args.model, args.dataset, args.wd, args.lr, args.saveID),'w') as f:
                            print(config['best_c'], file = f)

                    break
            
            # test ood/id
            print('#loading best models at epoch {} with c= {}'.format(config['best_epoch'],config['best_c']))
            saver.restore(sess, config['best_name'])
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

        # pretrain
        else:
            print('#load existing models.')

            model_file = 'mf_addressa_checkpoint/wd_1e-05_lr_0.001_mfnormalbce/669_ckpt.ckpt'
            saver.restore(sess, model_file)
            ret = test(sess, data, model, valid_set="test")
            perf_str = ' recall={}, ' \
                                    'precision={}, hit={}, ndcg={}'.format(str(ret["recall"]),
                                        str(ret['precision']), str(ret['hit_ratio']), str(ret['ndcg']))
            print(perf_str)
            exit()

            best_epoch = 199
            model_file = 'mf_addressa_checkpoint/wd_1e-05_lr_0.001_mf3branch_beta0/199_ckpt.ckpt'
            saver.restore(sess, model_file)
            c_best = 33.3
            for c in [0, c_best]:
                model.update_c(sess, c)
                ret = test(sess, data, model, model_type="rubi_both", valid_set="test")
                perf_str = 'c:{}: recall={}, ' \
                                        'precision={}, hit={}, ndcg={}'.format(c, str(ret["recall"]),
                                            str(ret['precision']), str(ret['hit_ratio']), str(ret['ndcg']))
                print(perf_str)
            exit()
            # pop_item_test = np.array([len(data.test_item_list.get(i, [])) for i in range(ITEM_NUM)])
            # users_to_test = list(data.test_user_list.keys())
            # for pop_exp in np.arange(1,10,1):
            #     # pop_exp = np.power(10,pop_exp)
            #     ret = test(sess, model, users_to_test, item_pop_test = pop_item_test, pop_exp=pop_exp, model_type="item_pop_test", valid_set="test")
            #     perf_str = 'pop_exp:%f  recall=[%.5f, %.5f], ' \
            #                         'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
            #                         (pop_exp, ret['recall'][0], ret['recall'][-1],
            #                             ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
            #                             ret['ndcg'][0], ret['ndcg'][-1])
            #     print(perf_str)
            #     perf_write = 'pop_exp:%f  %.4f %.4f %.4f' % \
            #                         (pop_exp, ret['hit_ratio'][0], ret['recall'][0],  ret['ndcg'][0])
            #     with open("result.txt","a+") as f:
            #         f.write(perf_write+"\n")

            
    # [94, 99, 129, 184, 249, 299]

    # print(config['best_c_epoch'], config['best_c_hr'], config['best_c_ndcg'], config['best_c_recall'], config['best_c_precision'])

    if args.early_stop == 0 and args.pretrain == 0 and args.test=='normal':
        print("{} dataset best epoch{}: hr:{} ndcg:{} recall:{} precision:{}".format(args.dataset, config['best_epoch'],config['best_hr'],config['best_ndcg'], config['best_recall'], config['best_pre']))
        logging.info("{} dataset best epoch{}: hr:{} ndcg:{} recall:{} precision:{}".format(args.dataset, config['best_epoch'],config['best_hr'],config['best_ndcg'], config['best_recall'], config['best_pre']))

        with open('{}_{}_checkpoint/wd_{}_lr_{}_{}/best_epoch.txt'.format(args.model, args.dataset, args.wd, args.lr, args.saveID),'w') as f:
            print(config['best_epoch'], file = f)

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    exit()