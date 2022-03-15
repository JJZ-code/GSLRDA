# coding:utf8
from random import shuffle, choice
from util.relation import Relation
from util.config import Config
from util.io import FileIO
import tensorflow as tf
from util import config
import numpy as np
import scipy.sparse as sp
import random


class GSLRDA:
    def __init__(self,conf,trainingSet=None,testSet=None):
        self.config = conf
        self.data = Relation(self.config, trainingSet, testSet)
        self.num_ncRNAs, self.num_drugs, self.train_size = self.data.trainingSize()
        self.emb_size = int(self.config['num.factors'])
        self.maxIter = int(self.config['num.max.iter'])
        learningRate = config.LineConfig(self.config['learnRate'])
        self.lRate = float(learningRate['-init'])
        regular = config.LineConfig(self.config['reg.lambda'])
        self.regU, self.regI, self.regB = float(regular['-u']), float(regular['-i']), float(regular['-b'])
        self.batch_size = int(self.config['batch_size'])
        self.u_idx = tf.placeholder(tf.int32, name="u_idx")
        self.v_idx = tf.placeholder(tf.int32, name="v_idx")
        self.r = tf.placeholder(tf.float32, name="rating")
        self.ncRNA_embeddings = tf.Variable(tf.truncated_normal(shape=[self.num_ncRNAs, self.emb_size], stddev=0.005),
                                             name='U')
        self.drug_embeddings = tf.Variable(tf.truncated_normal(shape=[self.num_drugs, self.emb_size], stddev=0.005),
                                           name='V')
        self.u_embedding = tf.nn.embedding_lookup(self.ncRNA_embeddings, self.u_idx)
        self.v_embedding = tf.nn.embedding_lookup(self.drug_embeddings, self.v_idx)
        config1 = tf.ConfigProto()
        config1.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config1)
        self.loss, self.lastLoss = 0, 0
        args = config.LineConfig(self.config['SGL'])
        self.ssl_reg = float(args['-lambda'])
        self.drop_rate = float(args['-droprate'])
        self.aug_type = int(args['-augtype'])
        self.ssl_temp = float(args['-temp'])
        norm_adj = self._create_adj_mat(is_subgraph=False)
        norm_adj = self._convert_sp_mat_to_sp_tensor(norm_adj)
        ego_embeddings = tf.concat([self.ncRNA_embeddings, self.drug_embeddings], axis=0)
        s1_embeddings = ego_embeddings
        s2_embeddings = ego_embeddings
        all_s1_embeddings = [s1_embeddings]
        all_s2_embeddings = [s2_embeddings]
        all_embeddings = [ego_embeddings]
        self.n_layers = 2
        self._create_variable()
        for k in range(0, self.n_layers):
            if self.aug_type in [0, 1]:
                self.sub_mat['sub_mat_1%d' % k] = tf.SparseTensor(
                    self.sub_mat['adj_indices_sub1'],
                    self.sub_mat['adj_values_sub1'],
                    self.sub_mat['adj_shape_sub1'])
                self.sub_mat['sub_mat_2%d' % k] = tf.SparseTensor(
                    self.sub_mat['adj_indices_sub2'],
                    self.sub_mat['adj_values_sub2'],
                    self.sub_mat['adj_shape_sub2'])
            else:
                self.sub_mat['sub_mat_1%d' % k] = tf.SparseTensor(
                    self.sub_mat['adj_indices_sub1%d' % k],
                    self.sub_mat['adj_values_sub1%d' % k],
                    self.sub_mat['adj_shape_sub1%d' % k])
                self.sub_mat['sub_mat_2%d' % k] = tf.SparseTensor(
                    self.sub_mat['adj_indices_sub2%d' % k],
                    self.sub_mat['adj_values_sub2%d' % k],
                    self.sub_mat['adj_shape_sub2%d' % k])

        #s1 - view
        for k in range(self.n_layers):
            s1_embeddings = tf.sparse_tensor_dense_matmul(self.sub_mat['sub_mat_1%d' % k],s1_embeddings)
            all_s1_embeddings += [s1_embeddings]
        all_s1_embeddings = tf.stack(all_s1_embeddings, 1)
        all_s1_embeddings = tf.reduce_mean(all_s1_embeddings, axis=1, keepdims=False)
        self.s1_ncRNA_embeddings, self.s1_drug_embeddings = tf.split(all_s1_embeddings, [self.num_ncRNAs, self.num_drugs], 0)

        #s2 - view
        for k in range(self.n_layers):
            s2_embeddings = tf.sparse_tensor_dense_matmul(self.sub_mat['sub_mat_2%d' % k],s2_embeddings)
            all_s2_embeddings += [s2_embeddings]
        all_s2_embeddings = tf.stack(all_s2_embeddings, 1)
        all_s2_embeddings = tf.reduce_mean(all_s2_embeddings, axis=1, keepdims=False)
        self.s2_ncRNA_embeddings, self.s2_drug_embeddings = tf.split(all_s2_embeddings, [self.num_ncRNAs, self.num_drugs], 0)


        for k in range(self.n_layers):
            ego_embeddings = tf.sparse_tensor_dense_matmul(norm_adj,ego_embeddings)

            all_embeddings += [ego_embeddings]
        all_embeddings = tf.stack(all_embeddings, 1)
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)

        self.main_ncRNA_embeddings, self.main_drug_embeddings = tf.split(all_embeddings, [self.num_ncRNAs, self.num_drugs], 0)

        self.neg_idx = tf.placeholder(tf.int32, name="neg_holder")
        self.neg_drug_embedding = tf.nn.embedding_lookup(self.main_drug_embeddings, self.neg_idx)
        self.u_embedding = tf.nn.embedding_lookup(self.main_ncRNA_embeddings, self.u_idx)
        self.v_embedding = tf.nn.embedding_lookup(self.main_drug_embeddings, self.v_idx)
        self.test = tf.reduce_sum(tf.multiply(self.u_embedding,self.main_drug_embeddings),1)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _convert_csr_to_sparse_tensor_inputs(self, X):
        coo = X.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        return indices, coo.data, coo.shape

    def _create_variable(self):
        self.sub_mat = {}
        if self.aug_type in [0, 1]:
            self.sub_mat['adj_values_sub1'] = tf.placeholder(tf.float32)
            self.sub_mat['adj_indices_sub1'] = tf.placeholder(tf.int64)
            self.sub_mat['adj_shape_sub1'] = tf.placeholder(tf.int64)

            self.sub_mat['adj_values_sub2'] = tf.placeholder(tf.float32)
            self.sub_mat['adj_indices_sub2'] = tf.placeholder(tf.int64)
            self.sub_mat['adj_shape_sub2'] = tf.placeholder(tf.int64)
        else:
            for k in range(self.n_layers):
                self.sub_mat['adj_values_sub1%d' % k] = tf.placeholder(tf.float32, name='adj_values_sub1%d' % k)
                self.sub_mat['adj_indices_sub1%d' % k] = tf.placeholder(tf.int64, name='adj_indices_sub1%d' % k)
                self.sub_mat['adj_shape_sub1%d' % k] = tf.placeholder(tf.int64, name='adj_shape_sub1%d' % k)

                self.sub_mat['adj_values_sub2%d' % k] = tf.placeholder(tf.float32, name='adj_values_sub2%d' % k)
                self.sub_mat['adj_indices_sub2%d' % k] = tf.placeholder(tf.int64, name='adj_indices_sub2%d' % k)
                self.sub_mat['adj_shape_sub2%d' % k] = tf.placeholder(tf.int64, name='adj_shape_sub2%d' % k)

    def _create_adj_mat(self, is_subgraph=False, aug_type=0):
        n_nodes = self.num_ncRNAs + self.num_drugs
        row_idx = [self.data.ncRNA[pair[0]] for pair in self.data.trainingData]
        col_idx = [self.data.drug[pair[1]] for pair in self.data.trainingData]
        if is_subgraph and aug_type in [0, 1, 2] and self.drop_rate > 0:
            # data augmentation type --- 0: Node Dropout; 1: Edge Dropout; 2: Random Walk
            if aug_type == 0:
                drop_ncRNA_idx = random.sample(list(range(self.num_ncRNAs)), int(self.num_ncRNAs * self.drop_rate))
                drop_drug_idx = random.sample(list(range(self.num_drugs)), int(self.num_drugs * self.drop_rate))
                indicator_ncRNA = np.ones(self.num_ncRNAs, dtype=np.float32)
                indicator_drug = np.ones(self.num_drugs, dtype=np.float32)
                indicator_ncRNA[drop_ncRNA_idx] = 0.
                indicator_drug[drop_drug_idx] = 0.
                diag_indicator_ncRNA = sp.diags(indicator_ncRNA)
                diag_indicator_drug = sp.diags(indicator_drug)
                R = sp.csr_matrix(
                    (np.ones_like(row_idx, dtype=np.float32), (row_idx, col_idx)),
                    shape=(self.num_ncRNAs, self.num_drugs))
                R_prime = diag_indicator_ncRNA.dot(R).dot(diag_indicator_drug)
                (ncRNA_np_keep, drug_np_keep) = R_prime.nonzero()
                ratings_keep = R_prime.data
                tmp_adj = sp.csr_matrix((ratings_keep, (ncRNA_np_keep, drug_np_keep+self.num_ncRNAs)), shape=(n_nodes, n_nodes))
            if aug_type in [1, 2]:
                keep_idx = random.sample(list(range(self.data.elemCount())), int(self.data.elemCount() * (1 - self.drop_rate)))
                ncRNA_np = np.array(row_idx)[keep_idx]
                drug_np = np.array(col_idx)[keep_idx]
                ratings = np.ones_like(ncRNA_np, dtype=np.float32)
                tmp_adj = sp.csr_matrix((ratings, (ncRNA_np, drug_np+self.num_ncRNAs)), shape=(n_nodes, n_nodes))
        else:
            ncRNA_np = np.array(row_idx)
            drug_np = np.array(col_idx)
            ratings = np.ones_like(ncRNA_np, dtype=np.float32)
            tmp_adj = sp.csr_matrix((ratings, (ncRNA_np, drug_np+self.num_ncRNAs)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        return adj_matrix

    def calc_ssl_loss(self):
        ncRNA_emb1 = tf.nn.embedding_lookup(self.s1_ncRNA_embeddings, tf.unique(self.u_idx)[0])
        ncRNA_emb2 = tf.nn.embedding_lookup(self.s2_ncRNA_embeddings, tf.unique(self.u_idx)[0])
        normalize_ncRNA_emb1 = tf.nn.l2_normalize(ncRNA_emb1, 1)
        normalize_ncRNA_emb2 = tf.nn.l2_normalize(ncRNA_emb2, 1)


        drug_emb1 = tf.nn.embedding_lookup(self.s1_drug_embeddings, tf.unique(self.v_idx)[0])
        drug_emb2 = tf.nn.embedding_lookup(self.s2_drug_embeddings, tf.unique(self.v_idx)[0])
        normalize_drug_emb1 = tf.nn.l2_normalize(drug_emb1, 1)
        normalize_drug_emb2 = tf.nn.l2_normalize(drug_emb2, 1)

        normalize_ncRNA_emb2_neg = normalize_ncRNA_emb2
        normalize_drug_emb2_neg = normalize_drug_emb2

        pos_score_ncRNA = tf.reduce_sum(tf.multiply(normalize_ncRNA_emb1, normalize_ncRNA_emb2), axis=1)
        ttl_score_ncRNA = tf.matmul(normalize_ncRNA_emb1, normalize_ncRNA_emb2_neg, transpose_a=False, transpose_b=True)

        pos_score_drug = tf.reduce_sum(tf.multiply(normalize_drug_emb1, normalize_drug_emb2), axis=1)
        ttl_score_drug = tf.matmul(normalize_drug_emb1, normalize_drug_emb2_neg, transpose_a=False, transpose_b=True)

        pos_score_ncRNA = tf.exp(pos_score_ncRNA / self.ssl_temp)
        ttl_score_ncRNA = tf.reduce_sum(tf.exp(ttl_score_ncRNA / self.ssl_temp), axis=1)
        pos_score_drug = tf.exp(pos_score_drug / self.ssl_temp)
        ttl_score_drug = tf.reduce_sum(tf.exp(ttl_score_drug / self.ssl_temp), axis=1)

        ssl_loss_ncRNA = -tf.reduce_sum(tf.log(pos_score_ncRNA / ttl_score_ncRNA)+1e-8)
        ssl_loss_drug = -tf.reduce_sum(tf.log(pos_score_drug / ttl_score_drug)+1e-8)
        ssl_loss = self.ssl_reg*(ssl_loss_ncRNA + ssl_loss_drug)
        return ssl_loss

    def calc_ssl_loss_v3(self):

        ncRNA_emb1 = tf.nn.embedding_lookup(self.s1_ncRNA_embeddings, tf.unique(self.u_idx)[0])
        ncRNA_emb2 = tf.nn.embedding_lookup(self.s2_ncRNA_embeddings, tf.unique(self.u_idx)[0])

        drug_emb1 = tf.nn.embedding_lookup(self.s1_drug_embeddings, tf.unique(self.v_idx)[0])
        drug_emb2 = tf.nn.embedding_lookup(self.s2_drug_embeddings, tf.unique(self.v_idx)[0])

        emb_merge1 = tf.concat([ncRNA_emb1, drug_emb1], axis=0)
        emb_merge2 = tf.concat([ncRNA_emb2, drug_emb2], axis=0)


        normalize_emb_merge1 = tf.nn.l2_normalize(emb_merge1, 1)
        normalize_emb_merge2 = tf.nn.l2_normalize(emb_merge2, 1)

        pos_score = tf.reduce_sum(tf.multiply(normalize_emb_merge1, normalize_emb_merge2), axis=1)
        ttl_score = tf.matmul(normalize_emb_merge1, normalize_emb_merge2, transpose_a=False, transpose_b=True)

        pos_score = tf.exp(pos_score / self.ssl_temp)
        ttl_score = tf.reduce_sum(tf.exp(ttl_score / self.ssl_temp), axis=1)
        ssl_loss = -tf.reduce_sum(tf.log(pos_score / ttl_score))
        ssl_loss = self.ssl_reg * ssl_loss
        return ssl_loss

    def next_batch_pairwise(self):
        shuffle(self.data.trainingData)
        batch_id = 0
        while batch_id < self.train_size:
            if batch_id + self.batch_size <= self.train_size:
                ncRNAs = [self.data.trainingData[idx][0] for idx in range(batch_id, self.batch_size + batch_id)]
                drugs = [self.data.trainingData[idx][1] for idx in range(batch_id, self.batch_size + batch_id)]
                batch_id += self.batch_size
            else:
                ncRNAs = [self.data.trainingData[idx][0] for idx in range(batch_id, self.train_size)]
                drugs = [self.data.trainingData[idx][1] for idx in range(batch_id, self.train_size)]
                batch_id = self.train_size

            u_idx, i_idx, j_idx = [], [], []
            drug_list = list(self.data.drug.keys())
            for i, ncRNA in enumerate(ncRNAs):
                i_idx.append(self.data.drug[drugs[i]])
                u_idx.append(self.data.ncRNA[ncRNA])
                neg_drug = choice(drug_list)
                while neg_drug in self.data.trainSet_u[ncRNA]:
                    neg_drug = choice(drug_list)
                j_idx.append(self.data.drug[neg_drug])

            yield u_idx, i_idx, j_idx

    def buildModel(self):
        y = tf.reduce_sum(tf.multiply(self.u_embedding, self.v_embedding), 1) \
            - tf.reduce_sum(tf.multiply(self.u_embedding, self.neg_drug_embedding), 1)
        rec_loss = -tf.reduce_sum(tf.log(tf.sigmoid(y))) + self.regU * (tf.nn.l2_loss(self.u_embedding) +
                                                                    tf.nn.l2_loss(self.v_embedding) +
                                                                    tf.nn.l2_loss(self.neg_drug_embedding))
        ssl_loss = self.calc_ssl_loss_v3()
        total_loss = rec_loss+ssl_loss

        opt = tf.train.AdamOptimizer(self.lRate)
        train = opt.minimize(total_loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)
        for iteration in range(self.maxIter):
            sub_mat = {}
            if self.aug_type in [0, 1]:
                sub_mat['adj_indices_sub1'], sub_mat['adj_values_sub1'], sub_mat[
                    'adj_shape_sub1'] = self._convert_csr_to_sparse_tensor_inputs(
                    self._create_adj_mat(is_subgraph=True, aug_type=self.aug_type))

                sub_mat['adj_indices_sub2'], sub_mat['adj_values_sub2'], sub_mat[
                    'adj_shape_sub2'] = self._convert_csr_to_sparse_tensor_inputs(
                    self._create_adj_mat(is_subgraph=True, aug_type=self.aug_type))
            else:
                for k in range(1, self.n_layers + 1):
                    sub_mat['adj_indices_sub1%d' % k], sub_mat['adj_values_sub1%d' % k], sub_mat[
                        'adj_shape_sub1%d' % k] = self._convert_csr_to_sparse_tensor_inputs(
                        self._create_adj_mat(is_subgraph=True, aug_type=self.aug_type))
                    sub_mat['adj_indices_sub2%d' % k], sub_mat['adj_values_sub2%d' % k], sub_mat[
                        'adj_shape_sub2%d' % k] = self._convert_csr_to_sparse_tensor_inputs(
                        self._create_adj_mat(is_subgraph=True, aug_type=self.aug_type))

            for n, batch in enumerate(self.next_batch_pairwise()):
                ncRNA_idx, i_idx, j_idx = batch
                feed_dict = {self.u_idx: ncRNA_idx,
                             self.v_idx: i_idx,
                             self.neg_idx: j_idx, }
                if self.aug_type in [0, 1]:
                    feed_dict.update({
                        self.sub_mat['adj_values_sub1']: sub_mat['adj_values_sub1'],
                        self.sub_mat['adj_indices_sub1']: sub_mat['adj_indices_sub1'],
                        self.sub_mat['adj_shape_sub1']: sub_mat['adj_shape_sub1'],
                        self.sub_mat['adj_values_sub2']: sub_mat['adj_values_sub2'],
                        self.sub_mat['adj_indices_sub2']: sub_mat['adj_indices_sub2'],
                        self.sub_mat['adj_shape_sub2']: sub_mat['adj_shape_sub2']
                    })
                else:
                    for k in range(self.n_layers):
                        feed_dict.update({
                            self.sub_mat['adj_values_sub1%d' % k]: sub_mat['adj_values_sub1%d' % k],
                            self.sub_mat['adj_indices_sub1%d' % k]: sub_mat['adj_indices_sub1%d' % k],
                            self.sub_mat['adj_shape_sub1%d' % k]: sub_mat['adj_shape_sub1%d' % k],
                            self.sub_mat['adj_values_sub2%d' % k]: sub_mat['adj_values_sub2%d' % k],
                            self.sub_mat['adj_indices_sub2%d' % k]: sub_mat['adj_indices_sub2%d' % k],
                            self.sub_mat['adj_shape_sub2%d' % k]: sub_mat['adj_shape_sub2%d' % k]
                        })

                _, l,rec_l,ssl_l = self.sess.run([train, total_loss, rec_loss, ssl_loss],feed_dict=feed_dict)
                print('training:', iteration + 1, 'batch', n, 'rec_loss:', rec_l, 'ssl_loss',ssl_l)





if __name__ == '__main__':
    conf = Config('GSLRDA.conf')
    for i in range(0, 1):
        train_path = f"./dataset/rtrain_{i}.txt"
        test_path = f"./dataset/rtest_{i}.txt"
        trainingData = FileIO.loadDataSet(conf, train_path, binarized=False, threshold=0)
        testData = FileIO.loadDataSet(conf, test_path, bTest=True, binarized=False,
                                      threshold=0)
        re = GSLRDA(conf, trainingData, testData)
        re.buildModel()
