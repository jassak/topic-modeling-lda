#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 18 June 2018

@author: jason
"""

import logging
import numbers
import numpy as np
import six
from collections import deque
import random

from gensim import utils, matutils
from abc_topicmodel import ABCTopicModel
from useful_datatypes import SparseCounter, SparseVector, SparseGraph
from aliassampler import AliasSampler

logger = logging.getLogger(__name__)

DTYPE_TO_EPS = {
    np.float16: 1e-5,
    np.float32: 1e-35,
    np.float64: 1e-100,
}


class LDAModelGrS(ABCTopicModel):
    """
    The constructor estimates Latent Dirichlet Allocation model parameters based
    on a training corpus, according to the Graph Sampler variant of
    Gibbs sampling method described in
    **Ahmed, Long, Silva, Wang: ﻿A Practical Algorithm for Solving the Incoherence
    Problem of Topic Models In Industrial Applications. KDD 2017**

    Example:
        lda = LDAModelGrS(corpus, num_topics=10)
    """

    def __init__(self, corpus=None, similarity_matrix=None, num_topics=100, alpha='symmetric', beta=None,
                 smooth_factor=0.1, num_passes=10, minimum_prob=0.01, random_state=None, dtype=np.float32):
        # TODO Comments
        """

        Args:
            corpus: If given, start training from the iterable `corpus` straight away. If not given,
                the model is left untrained (presumably because you want to call `train()` manually).

            similarity_matrix: ﻿stochastic matrix representing semantic similarity between words.
                Should be a numpy array in dense or sparse (scipy.sparse) format.

            num_topics: The number of requested latent topics to be extracted from
                the training corpus.

            alpha: Hyperparameter of the Dirichlet prior over the topics in a document.

            beta: Hyperparameter of the Dirichlet prior over the terms in a topic.

            smooth_factor: parameter controlling the influence of neighbour words.

            num_passes: The number of passes of the MCMC procedure. One pass is one step per term
                in each document of the whole corpus.

            minimum_prob: TODO

            random_state: TODO

            dtype: Data-type to use during calculations inside model. All inputs are also converted to this dtype.
                Available types: `numpy.float16`, `numpy.float32`, `numpy.float64`.
        """

        if dtype not in DTYPE_TO_EPS:
            raise ValueError(
                    "Incorrect 'dtype', please choose one of {}".format(
                            ", ".join("numpy.{}".format(tp.__name__) for tp in sorted(DTYPE_TO_EPS))))
        self.dtype = dtype

        logger.info("creating a new lda mhw model with {0} topics".format(num_topics))
        # store user-supplied parameters
        if corpus is not None:
            self.id2word = corpus.dictionary
            self.num_terms = 1 + max(self.id2word.keys())
        else:
            self.id2word = None
            self.num_terms = 0

        self.num_topics = int(num_topics)
        self.minimum_probability = minimum_prob
        self.random_state = utils.get_random_state(random_state)

        self.alpha, self.optimize_alpha = self.init_dir_prior(alpha, 'alpha')
        assert self.alpha.shape == (self.num_topics,), \
            "Invalid alpha shape. Got shape %s, but expected (%d, )" % (str(self.alpha.shape), self.num_topics)

        if isinstance(beta, six.string_types):
            if beta == 'asymmetric':
                raise ValueError("The 'asymmetric' option cannot be used for beta")
        self.beta, self.optimize_beta = self.init_dir_prior(beta, 'beta')
        assert self.beta.shape == (self.num_terms,) or self.beta.shape == (self.num_topics, self.num_terms), (
            "Invalid beta shape. Got shape %s, but expected (%d, 1) or (%d, %d)" %
            (str(self.beta.shape), self.num_terms, self.num_topics, self.num_terms))

        self.w_beta = sum(self.beta)

        self.term_seqs, self.topic_seqs, \
        self.doc_topic_counts, self.term_topic_counts, \
        self.terms_per_topic = \
            self.get_seqs_and_counts(corpus=corpus)
        self.num_docs = len(self.term_seqs)

        if similarity_matrix.shape != (self.num_terms, self.num_terms):
            raise ValueError("similarity_matrix must have shape (num_terms, num_terms)")
        if smooth_factor < 0 or smooth_factor > 1:
            raise ValueError("smooth_factor must be in [0, 1]")
        sim_graph, graph_aliassamplers = self.get_similarity_graph(similarity_matrix, smooth_factor)

        # if a training corpus was provided, start estimating the model right away
        if corpus is not None:
            self.train(corpus, sim_graph=sim_graph, graph_aliastab=graph_aliassamplers, num_passes=num_passes)
            self.theta, self.phi = self.get_theta_phi()

    def get_seqs_and_counts(self, corpus):
        """

        Args:
            corpus:

        Returns:

        """
        raise NotImplementedError

    def get_similarity_graph(self, similarity_matrix, smooth_factor):
        """

        Args:
            similarity_graph:
            smooth_factor: must be in range [0, 1]

        Returns:
            A useful_datatypes.SparseMatrix object corresponding to the matrix
            (1 - smooth_factor) * I + smooth_factor * similarity_matrix
        """
        eye = np.eye(self.num_terms, dtype=self.dtype)
        S = (1 - smooth_factor) * eye + smooth_factor * similarity_matrix
        sim_graph = SparseGraph(S, dtype=self.dtype)
        graph_aliassamplers = []
        for term_id in range(self.num_terms):
            neighbours = sim_graph.neighbours(term_id)
            weights = np.zeros(len(neighbours), self.dtype)
            for (idx, node) in enumerate(neighbours):
                weights[idx] = sim_graph[term_id, node]
            aliassampler = AliasSampler(weights, self.dtype)
            graph_aliassamplers.append((neighbours, aliassampler))
        return sim_graph, graph_aliassamplers

    def init_dir_prior(self, prior, name):
        # TODO move this method to the parent class.
        """
        Initializes the Dirichlet priors. Copied from gensim.

        Args:
            prior:
            name:

        Returns:

        """
        if prior is None:
            prior = 'symmetric'

        if name == 'alpha':
            prior_shape = self.num_topics
        elif name == 'beta':
            prior_shape = self.num_terms
        else:
            raise ValueError("'name' must be 'alpha' or 'beta'")

        is_auto = False

        # TODO Something is wrong here, I think it assigns beta = 1/num_topics for prior=symmetric
        if isinstance(prior, six.string_types):
            if prior == 'symmetric':
                logger.info("using symmetric %s at %s", name, 1.0 / self.num_topics)
                init_prior = np.asarray([1.0 / self.num_topics for _ in range(prior_shape)], dtype=self.dtype)
            elif prior == 'asymmetric':
                init_prior = \
                    np.asarray([1.0 / (i + np.sqrt(prior_shape)) for i in range(prior_shape)], dtype=self.dtype)
                init_prior /= init_prior.sum()
                logger.info("using asymmetric %s %s", name, list(init_prior))
            elif prior == 'auto':
                is_auto = True
                # This is obviously wrong since it's the same as symmetric. Maybe in future correct it.
                init_prior = np.asarray([1.0 / self.num_topics for _ in range(prior_shape)], dtype=self.dtype)
                if name == 'alpha':
                    logger.info("using autotuned %s, starting with %s", name, list(init_prior))
            else:
                raise ValueError("Unable to determine proper %s value given '%s'" % (name, prior))
        elif isinstance(prior, list):
            init_prior = np.asarray(prior, dtype=self.dtype)
        elif isinstance(prior, np.ndarray):
            init_prior = prior.astype(self.dtype, copy=False)
        elif isinstance(prior, np.number) or isinstance(prior, numbers.Real):
            init_prior = np.asarray([prior] * prior_shape, dtype=self.dtype)
        else:
            raise ValueError("%s must be either a np array of scalars, list of scalars, or scalar" % name)

        return init_prior, is_auto

    def train(self, corpus, sim_graph, graph_aliastab, num_passes=0):
        raise NotImplementedError

    def do_one_pass(self):
        raise NotImplementedError

    def sample_topics_for_one_doc(self):
        raise NotImplementedError

    def get_theta_phi(self):
        raise NotImplementedError
