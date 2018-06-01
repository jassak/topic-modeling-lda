#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 21 May 2018

@author: jason
"""

import logging
import numbers
import os

import numpy as np
import six
from scipy.special import gammaln, psi  # gamma function utils
from scipy.special import polygamma
from six.moves import xrange
from collections import defaultdict

from gensim import interfaces, utils, matutils
from gensim.matutils import (
    kullback_leibler, hellinger, jaccard_distance, jensen_shannon,
    dirichlet_expectation, logsumexp, mean_absolute_difference
)
from gensim.models import basemodel, CoherenceModel
from gensim.models.callbacks import Callback

logger = logging.getLogger('gensim.models.ldamodel')

DTYPE_TO_EPS = {
    np.float16: 1e-5,
    np.float32: 1e-35,
    np.float64: 1e-100,
}


class LDAModelCGS:
    # TODO Comments
    """

    """

    def __init__(self, corpus=None, num_topics=100, alpha='symmetric', beta=None, id2word=None, eval_every=10,
                 minimum_probability=0.01, random_state=None, dtype=np.float32):
        # TODO Comments
        """
        :param corpus:
        :param num_topics:
        :param id2word:
        :param eval_every:
        :param minimum_probability:
        :param random_state:
        :param dtype:
        """

        if dtype not in DTYPE_TO_EPS:
            raise ValueError(
                "Incorrect 'dtype', please choose one of {}".format(
                    ", ".join("numpy.{}".format(tp.__name__) for tp in sorted(DTYPE_TO_EPS))))
        self.dtype = dtype

        # store user-supplied parameters
        self.id2word = id2word
        if corpus is None and self.id2word is None:
            raise ValueError(
                'at least one of corpus/id2word must be specified, to establish input space dimensionality'
            )

        if self.id2word is None:
            logger.warning("no word id mapping provided; initializing from corpus, assuming identity")
            self.id2word = utils.dict_from_corpus(corpus)
            self.num_terms = len(self.id2word)
        elif len(self.id2word) > 0:
            self.num_terms = 1 + max(self.id2word.keys())
        else:
            self.num_terms = 0
        if self.num_terms == 0:
            raise ValueError("cannot compute LDA over an empty collection (no terms)")

        self.num_docs = len(corpus)
        self.num_topics = int(num_topics)
        self.minimum_probability = minimum_probability
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

        self.v_beta = sum(self.beta)

        self.term_seqs, self.topic_seqs, \
        self.doc_topic_counts, self.term_topic_counts, \
        self.terms_per_topic = \
            self.build_seqs_and_counts(corpus=corpus,id2word=self.id2word)

        # if a training corpus was provided, start estimating the model right away
        if corpus is not None:
           self.train(corpus)

    def build_seqs_and_counts(self, corpus, id2word):
        # TODO comments
        # Build term_seqs
        term_seqs = []
        for document in corpus:
            term_seq = []
            for term_pair in document:
                term_seq += [term_pair[0]] * int(term_pair[1])
            term_seqs.append(term_seq)
        # Init randomly topic_seqs
        topic_seqs = []
        for di in range(self.num_docs):
            topic_seq = np.random.randint(self.num_topics, size=len(term_seqs[di])).tolist()
            topic_seqs.append(topic_seq)
        # Build doc_topic_counts
        doc_topic_counts = []
        for topic_seq in topic_seqs:
            topic_count = [0] * self.num_topics
            for topic in topic_seq:
                topic_count[topic] += 1
            doc_topic_counts.append(topic_count)
        # Build term_topic_counts
        term_topic_counts = [None] * self.num_terms
        for term in range(self.num_terms):
            term_topic_counts[term] = [0] * self.num_topics
        for di in range(self.num_docs):
            assert len(term_seqs[di]) == len(topic_seqs[di]) # Check if everything is fine
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
        :param prior:
        :param name:
        :return:
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

    def train(self, corpus, eval_every=None, num_passes=1):
        # TODO Comments
        """
        :param corpus:
        :param eval_every:
        :param num_passes:
        :return:
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
        # logger.info(
        #     "running Gibbs Sampling LDA training, %s topics, over "
        #     "the supplied corpus of %i documents, evaluating perplexity every %i documents ",
        #     self.num_topics, lencorpus,
        #     eval_every
        # )

        # TODO start model training
        # Perform several rounds of Gibbs sampling on the documents in the given range.
        for _ in range(num_passes):
            self.do_one_pass()

    def do_one_pass(self):
        # One iteration of Gibbs sampling, across all documents.
        for di in range(self.num_docs):
            self.sample_topics_for_one_doc(di, self.term_seqs[di], self.topic_seqs[di], self.doc_topic_counts[di])

    def sample_topics_for_one_doc(self, di, one_doc_term_seq, one_doc_topic_seq,
                                  one_doc_topic_count):
        doc_len = len(one_doc_term_seq)
        num_topics = len(one_doc_topic_count)

        # Iterate over the positions (words) in the document
        for si in range(doc_len):
            term_id = one_doc_term_seq[si]
            old_topic = one_doc_topic_seq[si]

            # Remove this term from all counts
            one_doc_topic_count[old_topic] -= 1
            self.term_topic_counts[term_id][old_topic] -= 1
            self.terms_per_topic[old_topic] -= 1

            # Build a distribution over topics for this term
            topic_weights = np.zeros(num_topics, self.dtype)
            current_term_topic_count = self.term_topic_counts[term_id]
            for ti in range(num_topics):
                tw = ((current_term_topic_count[ti] + self.beta[term_id]) / (self.terms_per_topic[ti] + self.v_beta))\
                     * (one_doc_topic_count[ti] + self.alpha[ti])
                topic_weights[ti] = tw

            # Sample a topic assignment from this distribution
            topic_weights = topic_weights / sum(topic_weights)
            new_topic = np.random.choice(num_topics, p=topic_weights)

            # Put that new topic into the counts
            # TODO Important! This must change the something
            one_doc_topic_seq[si] = new_topic
            one_doc_topic_count[new_topic] += 1
            self.term_topic_counts[term_id][new_topic] += 1
            self.terms_per_topic[new_topic] += 1

        # Update seqs and counts class-wise
        self.topic_seqs[di] = one_doc_topic_seq
        self.doc_topic_counts[di] = one_doc_topic_count
