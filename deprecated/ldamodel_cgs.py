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

        # if a training corpus was provided, start estimating the model right away
        if corpus is not None:
            self.train(corpus, num_passes=num_passes)

    def init_seqs_and_counts(self, corpus):
        """
            Builds the sequences of terms and topics, and the counts of topics in docs,
            terms in topics and term per topic.
        Args:
            corpus:
        Returns:
            term_seqs, topic_seqs, doc_topic_counts, term_topic_counts, terms_per_topic
        """
        logger.info("initializing sequences and counts")
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
            topic_count = [0] * self.num_topics
            for topic in topic_seq:
                topic_count[topic] += 1
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
        self.term_seqs = term_seqs
        self.topic_seqs = topic_seqs
        self.doc_topic_counts = doc_topic_counts
        self.term_topic_counts = term_topic_counts
        self.terms_per_topic = terms_per_topic
        self.num_docs = len(self.term_seqs)

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
            logger.warning("LdaModel.train() called with an empty corpus")
            return

        # init sequences and counts
        self.init_seqs_and_counts(corpus=corpus)

        # Perform num_passes rounds of Gibbs sampling.
        logger.info(
                "running collapsed Gibbs sampling for LDA training, {0} topics, over "
                "the supplied corpus of {1} documents for {2} passes over the whole corpus"
                    .format(self.num_topics, lencorpus, num_passes)
        )
        for pass_i in range(num_passes):
            logger.info("gibbs sampling pass: {0}".format(pass_i))
            self.do_one_pass()
            # uncomment below for save-while-training
            self.save('models/model_cgs_currun_pass' + str(pass_i) + '.pkl')

        self.theta, self.phi = self.get_theta_phi()

    def do_one_pass(self):
        """
        Performs one iteration of Gibbs sampling, across all documents.
        """

        for doc_id in range(self.num_docs):
            if doc_id % 10 == 0:
                logger.info("doc: {0}".format(doc_id))
            else:
                logger.debug("doc: {0}".format(doc_id))
            self.sample_topics_for_one_doc(doc_id)

    def sample_topics_for_one_doc(self, doc_id):
        """
        Samples a sequence of topics by performing one pass of collapsed Gibbs sampling
        for one document, according to
        **﻿Griffiths, Steyvers: Finding ﻿scientific topics, PNAS 2004**
        Args:
            doc_id:
        """
        doc_term_seq = self.term_seqs[doc_id]
        doc_len = len(doc_term_seq)
        doc_topic_seq = self.topic_seqs[doc_id]
        doc_topic_count = self.doc_topic_counts[doc_id]
        num_topics = len(doc_topic_count)

        # Iterate over the positions (words) in the document
        for si in range(doc_len):
            term_id = doc_term_seq[si]
            old_topic = doc_topic_seq[si]
            logger.debug("sample topics for one doc iteration: position:{0}, term: {1}, old topic: {2}"
                         .format(si, term_id, old_topic))

            # Remove this topic from all counts
            doc_topic_count[old_topic] -= 1
            self.term_topic_counts[term_id][old_topic] -= 1
            self.terms_per_topic[old_topic] -= 1

            # Build a distribution over topics for this term
            topic_weights = np.zeros(num_topics, self.dtype)
            current_term_topic_count = self.term_topic_counts[term_id]
            for ti in range(num_topics):
                tw = ((current_term_topic_count[ti] + self.beta[term_id]) / (self.terms_per_topic[ti] + self.w_beta)) \
                     * (doc_topic_count[ti] + self.alpha[ti])
                topic_weights[ti] = tw
            topic_weights = topic_weights / sum(topic_weights)

            # Sample a topic assignment from this distribution
            new_topic = np.random.choice(num_topics, p=topic_weights)

            # Put that new topic into the counts
            doc_topic_seq[si] = new_topic
            doc_topic_count[new_topic] += 1
            self.term_topic_counts[term_id][new_topic] += 1
            self.terms_per_topic[new_topic] += 1

        # Update seqs and counts document-wise
        self.topic_seqs[doc_id] = doc_topic_seq
        self.doc_topic_counts[doc_id] = doc_topic_count

    def get_theta_phi(self):
        """
        Returns:
            theta and phi. Matrices whose vectors are the predictive distributions of
            topic|doc and term|topic respectively.
        """
        logger.info("computing theta and phi")
        theta = np.empty(shape=(self.num_docs, self.num_topics), dtype=self.dtype)
        phi = np.empty(shape=(self.num_topics, self.num_terms), dtype=self.dtype)

        for doc_id in range(self.num_docs):
            for topic_id in range(self.num_topics):
                theta[doc_id][topic_id] = self.doc_topic_counts[doc_id][topic_id] + self.alpha[topic_id]
            theta[doc_id] = theta[doc_id] / sum(theta[doc_id])

        for topic_id in range(self.num_topics):
            for term_id in range(self.num_terms):
                phi[topic_id][term_id] = self.term_topic_counts[term_id][topic_id] + self.beta[term_id]
            phi[topic_id] = phi[topic_id] / sum(phi[topic_id])

        return theta, phi

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