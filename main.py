#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
@author: jason
"""

def gen_erdosrenyi_graph(nnodes, avdeg):
    adj_mat = np.zeros((nnodes, nnodes), int)
    for i in range(nnodes):
        if not i % 100:
            logger.debug('erdos renyi, node:{0}'.format(i))
        for j in range(i + 1, nnodes):
            if random.random() < avdeg / (nnodes - 1):
                adj_mat[i][j] = 1
                adj_mat[j][i] = 1
    return adj_mat

def make_rand_sim_matrix(num_terms, avdeg):
    adj_mat = gen_erdosrenyi_graph(num_terms, avdeg)
    sim_mat = np.zeros((num_terms, num_terms), np.float32)
    for i in range(num_terms):
        if not i % 100:
            logger.debug('rand graph, node:{0}'.format(i))
        for j in range(i + 1, num_terms):
            if adj_mat[i][j] == 1:
                sim_mat[i][j] = random.random()
                sim_mat[j][i] = sim_mat[i][j]
    for i in range(num_terms):
        if not i % 100:
            logger.debug('rand graph normalization, node:{0}'.format(i))
        if sum(sim_mat[i]) > 0:
            sim_mat[i] = sim_mat[i] / sum(sim_mat[i])
    return sim_mat

def main():
    from ldamodel_cgs import LDAModelCGS
    from ldamodel_mhw import LDAModelMHW
    # from ldamodel_gs import LDAModelGrS
    from nipscorpus import NipsCorpus

    #==============#
    # QUICK TESTS: #
    #==============#
    # DO THIS FIRST FOR EVERY NEW MODEL:============================================#
    corpus = NipsCorpus()
    model = LDAModelMHW(corpus, num_topics=20)
    model.save('models/test_model.pkl')
    # THEN DO THIS:=================================================================#
    # model = LDAModelCGS.load('models/test_model.pkl')
    # model.do_one_pass()
    # stale_samples = {}
    #===============================================================================#


if __name__ == '__main__':
    import logging
    import numpy as np
    import random


    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename="logs/logger.log",
                        level=logging.INFO,
                        format=LOG_FORMAT
                        )

    logger = logging.getLogger(__name__)
    main()


