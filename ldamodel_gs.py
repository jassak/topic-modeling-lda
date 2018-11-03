#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 29 October 2018

@author: jason
"""

import logging
from abc_topicmodel import ABCTopicModel
from _ldags_train import train
from gensim import matutils

from profiling_utils import profileit

logger = logging.getLogger(__name__)


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

    def __init__(self, corpus=None, similarity_matrix=None, num_topics=100, smooth_factor=0.1, num_passes=10,
                 minimum_prob=0.01):

        """

        Args:
            corpus: If given, start training from the iterable `corpus` straight away. If not given,
                the model is left untrained (presumably because you want to call `train()` manually).

            similarity_matrix: ﻿stochastic matrix representing semantic similarity between words.
                Should be a numpy array in dense or sparse (scipy.sparse) format.

            num_topics: The number of requested latent topics to be extracted from
                the training corpus.

            smooth_factor: parameter controlling the influence of neighbour words.

            num_passes: The number of passes of the MCMC procedure. One pass is one step per term
                in each document of the whole corpus.

            minimum_prob: TODO
        """

        logger.info("creating a new lda graph sampler model with {0} topics".format(num_topics))
        # store user-supplied parameters
        if corpus is not None:
            self.id2word = corpus.dictionary
            self.num_terms = 1 + max(self.id2word.keys())
        else:
            self.id2word = None
            self.num_terms = 0

        self.num_topics = int(num_topics)
        self.minimum_probability = minimum_prob

        # if a training corpus was provided, start estimating the model right away
        if corpus is not None:
            self.train(corpus, similarity_matrix, smooth_factor, num_passes)

    @profileit
    def train(self, corpus, similarity_matrix, smooth_factor, num_passes=100):
        """
        Trains the model by making num_passes Monte Carlo passes on the corpus.

        Args:
            corpus:
            num_passes:

        """

        # Perform num_passes rounds of Gibbs sampling.
        logger.info(
                "running Graph Sampler sampling for LDA training, {0} topics, over "
                "the supplied corpus for {1} passes"
                    .format(self.num_topics, num_passes)
        )

        self.theta, self.phi = train(self.num_topics, num_passes, corpus, similarity_matrix, smooth_factor)

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

    def get_document_topics(self, doc_id, minimum_prob=0, readable=True):
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
