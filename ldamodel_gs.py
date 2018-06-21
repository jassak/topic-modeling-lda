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
from sparse_datastruct import SparseCounter, SparseVector, SparseGraph
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
            self.train(corpus, similarity_matrix=similarity_matrix, smooth_factor=smooth_factor, num_passes=num_passes)
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
        logger.info("creating sequences and counts")
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

    def get_similarity_graph(self, similarity_matrix, smooth_factor):
        """

        Args:
            similarity_graph:
            smooth_factor: must be in range [0, 1]

        Returns:
            A sparse_datastruct.SparseGraph corresponding to the matrix:
            (1 - smooth_factor) * I + smooth_factor * similarity_matrix.
        """
        eye = np.eye(self.num_terms, dtype=self.dtype)
        S = (1 - smooth_factor) * eye + smooth_factor * similarity_matrix
        sim_graph = SparseGraph(S, dtype=self.dtype)
        return sim_graph

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

    def train(self, corpus, similarity_matrix, smooth_factor, num_passes=0):
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

        logger.info(
                "running Graph Sampler sampling for LDA training, {0} topics, over "
                "the supplied corpus of {1} documents for {2} passes over the whole corpus"
                    .format(self.num_topics, lencorpus, num_passes)
        )

        # Create similarity graph and node alias samplers
        logger.info("creating similarity graph")
        if similarity_matrix.shape != (self.num_terms, self.num_terms):
            raise ValueError("similarity_matrix must have shape (num_terms, num_terms)")
        if smooth_factor < 0 or smooth_factor > 1:
            raise ValueError("smooth_factor must be in [0, 1]")
        sim_graph = self.get_similarity_graph(similarity_matrix, smooth_factor)

        # Init stale_samples
        logger.info("generating stale samples")
        stale_samples = self.init_stale_sample(sim_graph)

        # Perform num_passes rounds of Gibbs sampling.
        for pass_i in range(num_passes):
            logger.info("gibbs sampling pass: {0}".format(pass_i))
            self.do_one_pass(stale_samples, sim_graph)

        # Delete stuff
        del stale_samples
        del sim_graph

    def do_one_pass(self, stale_samples, sim_graph):
        """
        Performs one iteration of Gibbs sampling, across all documents.

        """

        for doc_id in range(self.num_docs):
            if doc_id % 1 == 0:
                logger.info("doc: {0}".format(doc_id))
            else:
                logger.debug("doc: {0}".format(doc_id))
            self.sample_topics_for_one_doc(doc_id, stale_samples, sim_graph)

    def sample_topics_for_one_doc(self, doc_id, stale_samples, sim_graph):
        """

        Args:
            doc_id:
            stale_samples:
            sim_graph:

        Returns:

        """
        doc_term_seq = self.term_seqs[doc_id]
        doc_len = len(doc_term_seq)
        doc_topic_seq = self.topic_seqs[doc_id]
        doc_topic_count = self.doc_topic_counts[doc_id]

        # Iterate over positions in document
        for si in range(doc_len):
            term_id = doc_term_seq[si]
            old_topic = doc_topic_seq[si]
            logger.debug("sample topics for one doc iteration: position:{0}, term: {1}, old topic: {2}"
                         .format(si, term_id, old_topic))

            # Check if stale samples for term_id are exhausted and generate if needed
            # If that's the case recompute qwS_norm
            if not stale_samples[term_id][0]:
                self.generate_stale_samples(term_id, stale_samples, self.num_topics + round(sim_graph.avdeg) - 1,
                                            sim_graph)
            (sw, _, _, qwS_norm) = stale_samples[term_id]

            # Remove current topic from counts
            doc_topic_count.decr_count(old_topic)  # use decr_count method for SparseCounter object instead of -=1
            self.term_topic_counts[term_id][old_topic] -= 1
            self.terms_per_topic[old_topic] -= 1

            # Compute sparse component pdw^S and pdw_norm^S
            (pdwS, pdwS_norm) = self.compute_sparse_comp(term_id, sim_graph, doc_topic_count, num_neighb=5)

            # Draw from proposal distribution q(t, w, d)^S with bucket sampling
            new_topic = self.bucket_sampling(term_id, pdwS, pdwS_norm, stale_samples, qwS_norm, sim_graph)

            # Draw a neighbour of term_id as a shortcut to compute qwS_approx (see paper)
            term_neighbour = sim_graph.sample_neighbour(term_id)
            (_, qwS_approx, _, _) = stale_samples[term_neighbour]

            # Accept new_topic with prob_ratio according to Metropolis-Hastings
            prob_ratio = (doc_topic_count.get_count(new_topic) + self.alpha[new_topic]) \
                         / (doc_topic_count.get_count(old_topic) + self.alpha[old_topic]) \
                         * (self.term_topic_counts[term_id][new_topic] + self.beta[term_id]) \
                         / (self.term_topic_counts[term_id][old_topic] + self.beta[term_id]) \
                         * (self.terms_per_topic[old_topic] + self.w_beta) \
                         / (self.terms_per_topic[new_topic] + self.w_beta) \
                         * ((pdwS_norm * pdwS[old_topic]) + (qwS_norm * qwS_approx[old_topic])) \
                         / ((pdwS_norm * pdwS[new_topic]) + (qwS_norm * qwS_approx[new_topic]))
            if prob_ratio >= 1.0:
                accept = True
            else:
                accept = (random.random() < prob_ratio)

            # If move is accepted put new topic into seqs and counts
            if accept:
                logger.debug("new topic accepted: {0}".format(new_topic))
                doc_topic_seq[si] = new_topic
                doc_topic_count.incr_count(new_topic)
                self.term_topic_counts[term_id][new_topic] += 1
                self.terms_per_topic[new_topic] += 1
            # Else put back old topic
            else:
                logger.debug("new topic was not accepted")
                doc_topic_seq[si] = old_topic
                doc_topic_count.incr_count(old_topic)
                self.term_topic_counts[term_id][old_topic] += 1
                self.terms_per_topic[old_topic] += 1

        # Update seqs and counts document-wise
        self.topic_seqs[doc_id] = doc_topic_seq
        self.doc_topic_counts[doc_id] = doc_topic_count

    def compute_sparse_comp(self, term_id, sim_graph, doc_topic_count, num_neighb=5):
        """

        Args:
            term_id:
            sim_graph:
            doc_topic_count:
            num_neighb:

        Returns:

        """
        # TODO comments + logging
        # sample num_neighb nodes from graph alias sampler O(1)
        nodes = [0] * num_neighb
        for n in range(num_neighb):
            nodes[n] = sim_graph.sample_neighbour(term_id)
        # compute pdwS and pdwS_norm O(k_d)
        pdwS = SparseVector(self.num_topics, dtype=self.dtype)
        pdwS_norm = 0
        for topic_id in doc_topic_count:
            pdwS[topic_id] = doc_topic_count.get_count(topic_id) \
                             / (self.terms_per_topic[topic_id] + self.w_beta)
            sum = 0
            for node in nodes:
                sum += (self.term_topic_counts[node][topic_id] + self.beta[node])
            pdwS[topic_id] *= sum / num_neighb
            pdwS_norm += pdwS[topic_id]
        pdwS /= pdwS_norm
        return pdwS, pdwS_norm

    def bucket_sampling(self, term_id, pdwS, pdwS_norm, stale_samples, qwS_norm, sim_graph):
        """

        Args:
            term_id:
            pdwS:
            pdwS_norm:
            stale_samples:
            qwS_norm:
            sim_graph:

        Returns:

        """
        # TODO comments + logging

        # Determine by coin flip to draw from sparse or dense bucket
        if random.random() < pdwS_norm / (pdwS_norm + qwS_norm):
            # TODO check if correct: for topic_id in pdwS returns topic_ids? if not change in mhw as well
            # If in sparse bucket: Draw from pdw^S in ﻿O(k_d)
            num_nnztopics = pdwS.get_nnz()
            topic_indices = []
            topic_weights = np.empty(num_nnztopics, dtype=self.dtype)
            for topic_id, weight_id in zip(pdwS, range(num_nnztopics)):
                topic_indices.append(topic_id)
                topic_weights[weight_id] = pdwS[topic_id]
            new_topic_idx = np.random.choice(num_nnztopics, p=topic_weights)
            return topic_indices[new_topic_idx]
        else:
            # If in dense bucket:
            # Draw neighb from graph_aliassamplers[term_id] in O(1)
            neighb = sim_graph.sample_neighbour(term_id)
            # Check if stale samples for neighb are exhausted and generate if needed
            if not stale_samples[neighb][0]:
                self.generate_stale_samples(neighb, stale_samples, self.num_topics + round(sim_graph.avdeg) - 1,
                                            sim_graph)
            (sneighb, _, _, _) = stale_samples[neighb]
            # Draw topic from stale_samples[neighb] in O(1)
            return sneighb.pop()

    def init_stale_sample(self, sim_graph):
        """

        Args:
            sim_graph:

        Returns:

        """
        # TODO comments + logging

        # generate stale samples for all term_ids
        stale_samples = {}
        for term_id in range(self.num_terms):
            self.generate_stale_samples(term_id, stale_samples, self.num_topics + round(sim_graph.avdeg) - 1, sim_graph,
                                        update_qwSnorm=False)
        # compute all qwS_norm by sparse dot-product between sim_graph and qw_norm's
        qw_norm_vec = np.empty(self.num_terms)
        for term_id in range(self.num_terms):
            (_, _, qw_norm_vec[term_id], _) = stale_samples[term_id]
        qwS_norm = sim_graph.dot_vec(qw_norm_vec)
        # put qwS_norm's in stale_samples
        for term_id in range(self.num_terms):
            (sw, qw, qw_norm, _) = stale_samples[term_id]
            stale_samples[term_id] = (sw, qw, qw_norm, qwS_norm[term_id])
        del qw_norm_vec, qwS_norm
        return stale_samples

    def generate_stale_samples(self, term_id, stale_samples, num_samples, sim_graph, update_qwSnorm=True):
        """
        Computes dense component of topic conditional distr for term_id qw as well as it's normalization qw_norm,
        then computes num_samples samples using AliasSampler and stores them in sw.
        Finally, writes (sw, qw, qw_norm) in stale_samples.

        Args:
            term_id:
            stale_samples:

        """
        logger.debug("generate stale samples for term: {0}".format(term_id))
        # Compute dense component of conditional topic distribution (q_w in Li et al. 2014)
        qw = np.zeros(self.num_topics, self.dtype)
        for topic_id in range(self.num_topics):
            qw[topic_id] = self.alpha[topic_id] * (self.term_topic_counts[term_id][topic_id] + self.beta[term_id]) \
                           / (self.terms_per_topic[topic_id] + self.w_beta)
        qw_norm = sum(qw)
        qw = qw / qw_norm
        logger.debug("is qw normalized properly? answer: sum(qw) = {0}".format(sum(qw)))
        # Sample num_topics samples from above distribution using the alias method
        alias_sampler = AliasSampler(qw, self.dtype)
        sw = alias_sampler.generate(num_samples)
        del alias_sampler
        # if asked to do so update qwS_norm
        if update_qwSnorm:
            qw_norm_vec = np.empty(self.num_terms)
            logger.debug("allocating an empty vec is efficient but risky! make sure everything's OK")
            for neighb in sim_graph.neighbours[term_id]:
                (_, _, qw_norm_vec[neighb], _) = stale_samples[neighb]
            qwS_norm = sim_graph.row_dot_vec(qw_norm_vec)
            logger.debug("qwS_norm[{0}] = {1}".format(term_id, qwS_norm))
            del qw_norm_vec
            stale_samples[term_id] = (sw, qw, qw_norm, qwS_norm)
        else:
            stale_samples[term_id] = (sw, qw, qw_norm, None)

    def get_theta_phi(self):
        """

        Returns:

        """
        logger.info("computing theta and phi")
        theta = np.empty(shape=(self.num_docs, self.num_topics), dtype=self.dtype)
        phi = np.empty(shape=(self.num_topics, self.num_terms), dtype=self.dtype)

        for doc_id in range(self.num_docs):
            doc_topic_count = self.doc_topic_counts[doc_id]
            for topic_id in range(self.num_topics):
                theta[doc_id][topic_id] = doc_topic_count.get_count(topic_id) + self.alpha[topic_id]
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
