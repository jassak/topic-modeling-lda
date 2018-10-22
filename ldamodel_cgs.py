#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 21 May 2018

@author: jason
"""

import logging
import _ldacgs_train
import numpy as np

from gensim import utils, matutils
from abc_topicmodel import ABCTopicModel

logger = logging.getLogger(__name__)

DTYPE_TO_EPS = {
    np.float16: 1e-5,
    np.float32: 1e-35,
    np.float64: 1e-100,
}


class LDAModelCGS(ABCTopicModel):
    """
    The constructor estimates Latent Dirichlet Allocation model parameters based
    on a training corpus, according to the collapsed Gibbs sampling method described in
    **﻿Griffiths, Steyvers: Finding ﻿scientific topics, PNAS 2004**

    Example:
        lda = LDAModelCGS(corpus, num_topics=10)
    """

    def __init__(self, corpus=None, num_topics=100, alpha='symmetric', beta=None, num_passes=10,
                 minimum_prob=0.01, random_state=None, dtype=np.float32):
        # TODO FIX: doesn't work when instantiated without a corpus and then trained later
        """

        Args:
            corpus: If given, start training from the iterable `corpus` straight away. If not given,
                the model is left untrained (presumably because you want to call `train()` manually).

            num_topics: The number of requested latent topics to be extracted from
                the training corpus.

            alpha: Hyperparameter of the Dirichlet prior over the topics in a document.

            beta: Hyperparameter of the Dirichlet prior over the terms in a topic.

            num_passes: The number of passes of the MCMC procedure. One pass is one step per term
                in each document of the whole corpus.

            minimum_prob: Minimum probability required for an object (term, topic) to be displayed (TODO should
            remove this)

            random_state: TODO findout what is this

            dtype: Data-type to use during calculations inside model. All inputs are also converted to this dtype.
                Available types: `numpy.float16`, `numpy.float32`, `numpy.float64`.
        """

        if dtype not in DTYPE_TO_EPS:
            raise ValueError(
                    "Incorrect 'dtype', please choose one of {}".format(
                            ", ".join("numpy.{}".format(tp.__name__) for tp in sorted(DTYPE_TO_EPS))))
        self.dtype = dtype

        logger.info("creating a new lda collapsed gibbs sampling model with {0} topics".format(num_topics))
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

        # self.alpha, self.optimize_alpha = init_dir_prior(self.num_topics, self.num_terms, self.dtype, alpha, 'alpha')
        # assert self.alpha.shape == (self.num_topics,), \
        #     "Invalid alpha shape. Got shape %s, but expected (%d, )" % (str(self.alpha.shape), self.num_topics)
        # if isinstance(beta, six.string_types):
        #     if beta == 'asymmetric':
        #         raise ValueError("The 'asymmetric' option cannot be used for beta")
        # self.beta, self.optimize_beta = init_dir_prior(self.num_topics, self.num_terms, self.dtype, beta, 'beta')
        # assert self.beta.shape == (self.num_terms,) or self.beta.shape == (self.num_topics, self.num_terms), (
        #     "Invalid beta shape. Got shape %s, but expected (%d, 1) or (%d, %d)" %
        #     (str(self.beta.shape), self.num_terms, self.num_topics, self.num_terms))
        # self.w_beta = sum(self.beta)

        # if a training corpus was provided, start estimating the model right away
        if corpus is not None:
            self.train(corpus, num_passes=num_passes)

    # @profileit
    def train(self, corpus, num_passes=10):
        """
        Trains the model by making num_passes Monte Carlo passes on the corpus.

        Args:
            corpus:
            num_passes:

        """
        try:
            lencorpus = len(corpus)
        except Exception:
            logger.warning("input corpus stream has no len(); counting documents")
            lencorpus = sum(1 for _ in corpus)
        if lencorpus == 0:
            logger.warning("LdaModelCGS.train() called with an empty corpus")
            return

        # init sequences and counts
        # self.init_seqs_and_counts(corpus=corpus)

        # Perform num_passes rounds of Gibbs sampling.
        logger.info(
                "running collapsed Gibbs sampling for LDA training, {0} topics, over "
                "the supplied corpus of {1} documents for {2} passes over the whole corpus"
                    .format(self.num_topics, lencorpus, num_passes)
        )
        self.theta, self.phi = _ldacgs_train.train(self.num_topics, num_passes, corpus)

    def get_topic_terms(self, topic_id, topn=10, readable=True):
        # TODO move this and similar methods to parent class
        """

        Args:
            topic_id:
            topn:
            readable: If False returns term_id, if True returns the actual word.

        Returns:
             A list of tuples (term, prob) of the topn terms in topic_id, formated according to format.

        """

        topic_term_probs = self.phi[topic_id]
        bestn = matutils.argsort(topic_term_probs, topn, reverse=True)
        if readable:
            return [(self.id2word[idx], topic_term_probs[idx]) for idx in bestn]
        else:
            return [(idx, topic_term_probs[idx]) for idx in bestn]

    def get_term_topics(self, term_id, topn=10, minimum_prob=0):
        """

        Args:
            term_id:
            topn:
            minimum_prob:

        Returns:
            A list of tuples (topic, prob) of topics containing term_id with prob greater than minimum_prob.

        """

        term_topic_probs = self.phi.transpose()[term_id]
        sorted_probs = matutils.argsort(term_topic_probs, topn=topn, reverse=True)
        return [(topic_id, term_topic_probs[topic_id]) for topic_id in sorted_probs
                if term_topic_probs[topic_id] > minimum_prob]

    def get_document_topics(self, doc_id, minimum_prob=None, readable=True):
        """

        Args:
            doc_id:
            minimum_prob: Ignore topics below this probability.
            readable: If False returns topic_id's. Else returns a string of the top 10 words in topic.

        Returns:
            A list of tuples (topic, probability) for document[doc_id]. topic is either topic_id or a string.

        """

        if minimum_prob is None:
            minimum_prob = self.minimum_probability
        minimum_prob = max(minimum_prob, 1e-8)  # never allow zero values in sparse output. (Why??)

        doc_topic_probs = self.theta[doc_id]
        sorted_idx = matutils.argsort(doc_topic_probs, reverse=True)
        if not readable:
            return [(idx, doc_topic_probs[idx]) for idx in sorted_idx if doc_topic_probs[idx] > minimum_prob]
        else:
            doc_topics = []
            for idx in sorted_idx:
                if doc_topic_probs[idx] > minimum_prob:
                    topic_terms = self.get_topic_terms(idx, topn=10, readable=True)
                    terms_string = " ".join([w[0] for w in topic_terms])
                    doc_topics.append((terms_string, doc_topic_probs[idx]))
            return doc_topics

    def get_topic_documents(self, topic_id, topn=10, minimum_prob=0):
        """

        Args:
            topic_id:
            minimum_prob:

        Returns:
            A list of tuples (doc_id, probability) of documents containing topic_id with prob greater than minimum_prob.

        """

        topic_docs_probs = self.theta.transpose()[topic_id]
        sorted_probs = matutils.argsort(topic_docs_probs, topn=topn, reverse=True)
        return [(doc_id, topic_docs_probs[doc_id]) for doc_id in sorted_probs
                if topic_docs_probs[doc_id] > minimum_prob]
