#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TODO in LDAModelMHW:
    run profiler to see what's going on with the cython function
    consider cythonize the entire gen_stale_samples
    cythonize compute_sparse_comp and bucket_sampling
"""

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
    from ldamodel_mhw import LDAModelMHW
    from nipscorpus import NipsCorpus

    corpus = NipsCorpus()
    model = LDAModelMHW(corpus, num_topics=100, num_passes=5, dtype=np.float64)
    model.save('models/test_model_mhw_t100p5.pkl')


if __name__ == '__main__':
    import logging
    import numpy as np
    import random
    import cProfile

    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename="logs/logger.log",
                        level=logging.INFO,
                        format=LOG_FORMAT
                        )

    logger = logging.getLogger(__name__)
    main()
    # cProfile.run('main()')
