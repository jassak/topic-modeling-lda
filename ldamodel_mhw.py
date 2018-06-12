#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 11 June 2018

@author: jason
"""

import logging
import numbers
import numpy as np
import six

from gensim import utils, matutils
from abc_topicmodel import ABCTopicModel
from useful_datatypes import SparseCounter

logger = logging.getLogger(__name__)

DTYPE_TO_EPS = {
    np.float16: 1e-5,
    np.float32: 1e-35,
    np.float64: 1e-100,
}


class LDAModelMHW(ABCTopicModel):
    """
    The constructor estimates Latent Dirichlet Allocation model parameters based
    on a training corpus, according to the Metropolis-Hasting-Walker variant of
    Gibbs sampling method described in
    **Li, Ahmed, Ravi, Smola: ﻿Reducing the sampling complexity of topic models. ﻿KDD 2014.**

    Example:
        lda = LDAModelMHW(corpus, num_topics=10)
    """
    def __init__(self, corpus=None, num_topics=100, alpha='symmetric', beta=None, num_passes=10,
                 eval_every=10, minimum_prob=0.01, random_state=None, dtype=np.float32):
        # TODO Comments
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

            eval_every: TODO

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

        # store user-supplied parameters
        if corpus is not None:
            self.id2word = corpus.dictionary
            self.num_terms = 1 + max(self.id2word.keys())
        else:
            self.id2word = None
            self.num_terms = 0

        self.num_topics = int(num_topics)
        self.minimum_probability = minimum_prob
        self.eval_every = eval_every
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

        # if a training corpus was provided, start estimating the model right away
        if corpus is not None:
            self.train(corpus, num_passes=num_passes)
            self.theta, self.phi = self.get_theta_phi()

    def get_seqs_and_counts(self, corpus):
        """
            Builds the sequences of terms and topics, and the counts of topics in docs,
            terms in topics and term per topic.
            Important note: since the present model exploits document_topic_count sparsity,
            the corresponding counts are implemented using the custom datatype SparseCount.

        Args:
            corpus:

        Returns:
            term_seqs, topic_seqs, doc_topic_counts, term_topic_counts, terms_per_topic
        """
        # Build term_seqs
        term_seqs = []
        for document in corpus:
            term_seq = []
            for term_pair in document:
                term_seq += [term_pair[0]] * int(term_pair[1])
            term_seqs.append(term_seq)
        # Init randomly topic_seqs
        topic_seqs = []
        for di in range(len(term_seqs)):
            topic_seq = np.random.randint(self.num_topics, size=len(term_seqs[di])).tolist()
            topic_seqs.append(topic_seq)
        # Build doc_topic_counts
        doc_topic_counts = []
        for topic_seq in topic_seqs:
            topic_count = SparseCounter(topic_seq)
            doc_topic_counts.append(topic_count)
        # Build term_topic_counts
        term_topic_counts = [None] * self.num_terms
        for term in range(self.num_terms):
            term_topic_counts[term] = [0] * self.num_topics
        for di in range(len(term_seqs)):
            assert len(term_seqs[di]) == len(topic_seqs[di])  # Check if everything is fine
            for term, topic in zip(term_seqs[di], topic_seqs[di]):
                term_topic_counts[term][topic] += 1
        # Sum above across terms to build terms_per_topic
        terms_per_topic = [0] * self.num_topics
        for topic in range(self.num_topics):
            for term in range(self.num_terms):
                terms_per_topic[topic] += term_topic_counts[term][topic]
        return term_seqs, topic_seqs, doc_topic_counts, term_topic_counts, terms_per_topic

    def init_dir_prior(self, prior, name):
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

    def train(self, corpus, eval_every=None, num_passes=0):
        """
        Trains the model by making num_passes Monte Carlo passes on the corpus.

        Args:
            corpus:
            eval_every:
            num_passes:

        """

        if eval_every is None:
            eval_every = self.eval_every

        try:
            lencorpus = len(corpus)
        except Exception:
            logger.warning("input corpus stream has no len(); counting documents")
            lencorpus = sum(1 for _ in corpus)
        if lencorpus == 0:
            logger.warning("LdaModel.train() called with an empty corpus")
            return

        # TODO Write the correct version of the logger
        logger.info(
                "running Gibbs Sampling LDA training, %s topics, over "
                "the supplied corpus of %i documents, evaluating perplexity every %i documents ",
                self.num_topics, lencorpus,
                eval_every
        )

        # Perform several rounds of Gibbs sampling on the documents in the given range.
        print('Start training:')
        for pass_i in range(num_passes):
            print('\tpass', pass_i)
            self.do_one_pass()

    def do_one_pass(self):
        """
        Performs one iteration of Gibbs sampling, across all documents.

        """

        for doc_id in range(self.num_docs):
            self.sample_topics_for_one_doc(doc_id)

    def sample_topics_for_one_doc(self, doc_id):
        doc_term_seq = self.term_seqs[doc_id]
        doc_len = len(doc_term_seq)
        doc_topic_seq = self.topic_seqs[doc_id]
        doc_topic_count = self.doc_topic_counts[doc_id]
        num_topics = len(doc_topic_count)
        # TODO Continue here

    def get_theta_phi(self):
        pass

    def get_topic_terms(self, topic_id, topn=10, readable=True):
        pass

    def get_term_topics(self, term_id, topn=10, minimum_prob=0):
        pass

    def get_document_topics(self, doc_id, minimum_prob=0, readable=True):
        pass

    def get_topic_documents(self, topic_id, topn=10, minimum_prob=0):
        pass
