#!/usr/bin/env cython
# coding: utf-8

"""
Created on 19 October 2018

@author: jason
"""

# TODO cython decorators where needed (also in sparsecounter.pyx)

cimport cython
from libc.stdlib cimport malloc, free, rand, RAND_MAX
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.stdio cimport printf
import numpy as np
cimport numpy as np


def init_seqs_and_counts(num_topics, num_terms, corpus):
    # Build term_seqs[d][s]
    term_seqs = []
    for document in corpus:
        term_seq = []
        for term_pair in document:
            term_seq += [term_pair[0]] * int(term_pair[1])
        term_seqs.append(term_seq)
    # Init randomly topic_seqs[d][s]
    topic_seqs = []
    for docid in range(len(term_seqs)):
        topic_seq = np.random.randint(num_topics, size=len(term_seqs[docid])).tolist()
        topic_seqs.append(topic_seq)
    # Build doc_topic_counts[d][t]
    doc_topic_counts = []
    for topic_seq in topic_seqs:
        topic_count = [0] * num_topics
        for topic in topic_seq:
            topic_count[topic] += 1
        doc_topic_counts.append(topic_count)
#    # Build term_topic_counts[w][t]
    term_topic_counts = [None] * num_terms
    for term in range(num_terms):
        term_topic_counts[term] = [0] * num_topics
    for di in range(len(term_seqs)):
        assert len(term_seqs[di]) == len(topic_seqs[di])  # Check if everything is fine
        for term, topic in zip(term_seqs[di], topic_seqs[di]):
            term_topic_counts[term][topic] += 1
#    # Sum above across terms to build terms_per_topic[t]
    terms_per_topic = [0] * num_topics
    for topic in range(num_topics):
        for term in range(num_terms):
            terms_per_topic[topic] += term_topic_counts[term][topic]
    return term_seqs, topic_seqs, doc_topic_counts, term_topic_counts, terms_per_topic

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int ** _cythonize_2dlist(list lists):
    cdef int num_lists = len(lists)
    cdef int list_len
    cdef int i, j
    cdef int ** cLists
    # malloc
    cLists = <int **>PyMem_Malloc(num_lists * sizeof(int *))
    for i in range(num_lists):
        list_len = len(lists[i])
        cLists[i] = <int *>PyMem_Malloc(list_len * sizeof(int))
        # list to C pointer
        for j in range(list_len):
            cLists[i][j] = <int>lists[i][j]
    return cLists

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int * _cythonize_1dlist(list alist):
    cdef int list_len = len(alist)
    cdef int i
    cdef int * cList
    # malloc
    cList = <int *>PyMem_Malloc(list_len * sizeof(int))
    # list to C pointer
    for i in range(list_len):
        cList[i] = <int>alist[i]
    return cList

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double * _init_prior(char prior_type, int num_topics, int num_terms):
    cdef int i
    cdef double * prior
    if prior_type == 'a': # prior_type == alpha
        prior = <double *>PyMem_Malloc(num_topics * sizeof(double))
        for i in range(num_topics):
            prior[i] = 1.0 / num_topics
    elif prior_type == 'b': # prior_type == beta
        prior = <double *>PyMem_Malloc(num_terms * sizeof(double))
        for i in range(num_terms):
            prior[i] = 1.0 / num_terms
    else:
        raise ValueError("prior_type must be 'a' for alpha or 'b' for beta")
    return prior

def train(num_topics, num_passes, corpus):
    cdef int i, j
    cdef int num_docs
    cdef int num_terms
    cdef int ** cTermSeqs
    cdef int ** cTopicSeqs
    cdef int ** cDocTopicCounts
    cdef int ** cTermTopicCounts
    cdef int * cTermsPerTopic
    cdef int * doc_len
    cdef double * alpha
    cdef double * beta
    cdef double w_beta
    cdef list theta, phi
    cdef double sum
    # get num_terms from corpus
    id2word = corpus.dictionary
    num_terms = 1 + max(id2word.keys())
    del id2word

    # init sequences and counts
    term_seqs, topic_seqs, doc_topic_counts, term_topic_counts, terms_per_topic = init_seqs_and_counts(num_topics, num_terms, corpus)
    num_docs = len(term_seqs)

    # init doc_len
    doc_len = <int *>PyMem_Malloc(num_docs * sizeof(int))
    for i in range(num_docs):
        doc_len[i] = len(term_seqs[i])

    # convert lists to C pointers
    cTermSeqs = _cythonize_2dlist(term_seqs)
    cTopicSeqs = _cythonize_2dlist(topic_seqs)
    cDocTopicCounts = _cythonize_2dlist(doc_topic_counts)
    cTermTopicCounts = _cythonize_2dlist(term_topic_counts)
    cTermsPerTopic = _cythonize_1dlist(terms_per_topic)

    # init priors
    alpha = _init_prior('a', num_topics, num_terms)
    beta = _init_prior('b', num_topics, num_terms)
    w_beta = 0.
    for i in range(num_terms):
        w_beta += beta[i]

    # call cythonized train
    _train(num_topics, num_docs, doc_len, cTermSeqs, cTopicSeqs, cDocTopicCounts,
    cTermTopicCounts, cTermsPerTopic, alpha, beta, w_beta, num_passes)

    # allocate theta, phi
    theta = [None] * num_docs
    for i in range(num_docs):
        theta[i] = [0.0] * num_topics
    phi = [None] * num_topics
    for i in range(num_topics):
        phi[i] = [0.0] * num_terms

    # computer theta, phi
    for i in range(num_docs):
        sum = 0.0
        for j in range(num_topics):
            theta[i][j] = cDocTopicCounts[i][j] + alpha[j]
            sum += theta[i][j]
        for j in range(num_topics):
            theta[i][j] /= sum
    for i in range(num_topics):
        sum = 0.0
        for j in range(num_terms):
            phi[i][j] = cTermTopicCounts[j][i] + beta[j]
            sum += phi[i][j]
        for j in range(num_terms):
            phi[i][j] /= sum
    return theta, phi

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _train(int num_topics, int num_docs, int * doc_len,
                int ** cTermSeqs, int ** cTopicSeqs,
                int ** cDocTopicCounts, int ** cTermTopicCounts, int * cTermsPerTopic,
                double * alpha, double * beta, double w_beta,
                int num_passes):

    cdef int p, d, s, t
    cdef int cur_w
    cdef int old_t, new_t
    cdef double * topic_weights
    cdef double tw_sum

    # malloc
    topic_weights = <double *>PyMem_Malloc(num_topics * sizeof(double))

    # start monte carlo
    for p in range(num_passes):
        printf("pass: %d\n", p)
        for d in range(num_docs):
            for s in range(doc_len[d]):
                # Get current term and topic
                cur_w = cTermSeqs[d][s]
                old_t = cTopicSeqs[d][s]

                # Remove this topic from all counts
                cDocTopicCounts[d][old_t] -= 1
                cTermTopicCounts[cur_w][old_t] -= 1
                cTermsPerTopic[old_t] -= 1

                # Build a distribution over topics for this term
                tw_sum = 0.
                for t in range(num_topics):
                    topic_weights[t] = (cTermTopicCounts[cur_w][t] + beta[cur_w]) * (cDocTopicCounts[d][t] + alpha[t]) /(cTermsPerTopic[t] + w_beta)
                    tw_sum += topic_weights[t]
                # Normalize it
                for t in range(num_topics):
                    topic_weights[t] /= tw_sum

                # TODO test if calling pure C faster, possibly replace with a better rng
                new_t = rand_choice(num_topics, topic_weights)

                # Put that new topic into the counts
                cTopicSeqs[d][s] = new_t
                cDocTopicCounts[d][new_t] += 1
                cTermTopicCounts[cur_w][new_t] += 1
                cTermsPerTopic[new_t] += 1

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef int rand_choice(int n, double * prob) nogil:
    cdef int i
    cdef double r
    cdef double cuml
    r = <double> rand() / RAND_MAX
    cuml = 0.0
    for i in range(n):
        cuml = cuml + prob[i]
        if (r <= cuml):
            return i
