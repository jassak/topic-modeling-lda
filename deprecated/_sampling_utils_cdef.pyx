#!/usr/bin/env cython
# coding: utf-8
# cython: embedsignature=True

# TODO follow gensim: a python wraper using def calling a cython function using cdef
# TODO check if possible to remove doc_id from args

"""
Created on 5 August 2018

@author: jason
"""

cimport cython
import numpy as np
cimport numpy as np
ctypedef cython.floating DTYPE_t
from libc.stdlib cimport malloc, free

from cpython.mem cimport PyMem_Malloc, PyMem_Free

def cgs_do_one_pass(int num_docs, int num_topics,
                    DTYPE_t[:] alpha, DTYPE_t[:] beta, DTYPE_t w_beta,
                    list term_seqs, list topic_seqs,
                    list doc_topic_counts, list term_topic_counts, list terms_per_topic):
    """
    Performs one iteration of Gibbs sampling, across all documents.

    """

    # cdefs
    cdef int i
    cdef int j
    cdef int doc_id
    cdef int doc_len
    cdef int num_terms
    cdef int cur_dtc_len
    cdef int cur_terseq_len
    cdef int cur_topseq_len
    cdef int *cur_doc_topic_count
    cdef int *cur_term_seq
    cdef int *cur_topic_seq

    num_terms = len(term_topic_counts)

    # malloc cTerm_seqs
    cdef int ** cTerm_seqs
    cTerm_seqs = <int **>PyMem_Malloc(num_docs * sizeof(int *))
    for i in range(num_docs):
        cTerm_seqs[i] = <int *>PyMem_Malloc(len(term_seqs[i]) * sizeof(int))
    for i in range(num_docs):
        for j in range(len(term_seqs[i])):
            cTerm_seqs[i][j] = term_seqs[i][j]
    # malloc cTopic_seqs
    cdef int ** cTopic_seqs
    cTopic_seqs = <int **>PyMem_Malloc(num_docs * sizeof(int *))
    for i in range(num_docs):
        cTopic_seqs[i] = <int *>PyMem_Malloc(len(topic_seqs[i]) * sizeof(int))
    for i in range(num_docs):
        for j in range(len(topic_seqs[i])):
            cTopic_seqs[i][j] = topic_seqs[i][j]
    # malloc cDoc_topic_counts
    cdef int ** cDoc_topic_counts
    cDoc_topic_counts = <int **>PyMem_Malloc(num_docs * sizeof(int *))
    for i in range(num_docs):
        cDoc_topic_counts[i] = <int *>PyMem_Malloc(num_topics * sizeof(int))
    for i in range(num_docs):
        for j in range(num_topics):
            cDoc_topic_counts[i][j] = doc_topic_counts[i][j]
    # malloc cTerm_topic_counts
    cdef int ** cTerm_topic_counts
    cTerm_topic_counts = <int **>PyMem_Malloc(num_terms * sizeof(int *))
    for i in range(num_terms):
        cTerm_topic_counts[i] = <int *>PyMem_Malloc(num_topics * sizeof(int))
    for i in range(num_terms):
        for j in range(num_topics):
            cTerm_topic_counts[i][j] = term_topic_counts[i][j]
    # malloc cTerms_per_topic
    cdef int * cTerms_per_topic
    cTerms_per_topic = <int *>PyMem_Malloc(num_topics * sizeof(int))
    for i in range(num_topics):
        cTerms_per_topic[i] = terms_per_topic[i]

    for doc_id in range(num_docs):
        if doc_id % 50 == 0:
            print(doc_id)
        doc_len = len(term_seqs[doc_id])

#        # malloc cur_doc_topic_count
#        cur_dtc_len = len(doc_topic_counts[doc_id])
#        cur_doc_topic_count = <int *> malloc(cur_dtc_len * cython.sizeof(int))
#        if cur_doc_topic_count is NULL:
#            raise MemoryError()
#        for i in range(cur_dtc_len):
#            cur_doc_topic_count[i] = doc_topic_counts[doc_id][i]
#        # malloc cur_term_seq
#        cur_terseq_len = len(term_seqs[doc_id])
#        cur_term_seq = <int *> malloc(cur_terseq_len * cython.sizeof(int))
#        if cur_term_seq is NULL:
#            raise MemoryError()
#        for i in range(cur_terseq_len):
#            cur_term_seq[i] = term_seqs[doc_id][i]
#        # malloc cur_topic_seq
#        cur_topseq_len = len(topic_seqs[doc_id])
#        cur_topic_seq = <int *> malloc(cur_topseq_len * cython.sizeof(int))
#        if cur_topic_seq is NULL:
#            raise MemoryError()
#        for i in range(cur_topseq_len):
#            cur_topic_seq[i] = topic_seqs[doc_id][i]

        # call C function
        cgs_sample_topics_for_one_doc(doc_id, doc_len, num_topics, alpha, beta, w_beta,
                                        cTerm_seqs[doc_id], cTopic_seqs[doc_id],
                                        cDoc_topic_counts[doc_id], term_topic_counts, terms_per_topic)

#        # free memory
#        free(cur_doc_topic_count)
#        free(cur_term_seq)
#        free(cur_topic_seq)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void cgs_sample_topics_for_one_doc(int doc_id, int doc_len, int num_topics,
                                        DTYPE_t[:] alpha, DTYPE_t[:] beta, DTYPE_t w_beta,
                                        int * term_seq, int * topic_seq,
                                        int * cur_doc_topic_count, int ** term_topic_counts, int ** terms_per_topic):
    """
    Cython version:
    Samples a sequence of topics by performing one pass of collapsed Gibbs sampling
    for one document, according to
    **﻿Griffiths, Steyvers: Finding ﻿scientific topics, PNAS 2004**

    Args:
        doc_id:

    """

    # cdefs
    cdef int si
    cdef int ti
    cdef int term_id
    cdef int old_topic
    cdef int new_topic
    cdef double one_beta
    cdef double tw_sum
    cdef list topic_weights

    # Iterate over the positions (words) in the document
    for si in range(doc_len):
        term_id = term_seq[si]
        old_topic = topic_seq[si]

        # Remove this topic from all counts
        cur_doc_topic_count[old_topic] -= 1
        term_topic_counts[term_id][old_topic] -= 1
        terms_per_topic[old_topic] -= 1

        # localize some variables
        cur_term_topic_count = term_topic_counts[term_id]
        one_beta = beta[term_id]

        # Build a distribution over topics for this term
        topic_weights = [((cur_term_topic_count[ti] + one_beta)
                / (terms_per_topic[ti] + w_beta)
                * (cur_doc_topic_count[ti] + alpha[ti])) for ti in range(num_topics)]
        tw_sum = sum(topic_weights)
        topic_weights = [topic_weights[ti] / tw_sum for ti in range(num_topics)]

        # Sample a topic assignment from this distribution
        new_topic = np.random.choice(num_topics, p=topic_weights)

        # Put that new topic into the counts
        topic_seq[si] = new_topic
        cur_doc_topic_count[new_topic] += 1
        term_topic_counts[term_id][new_topic] += 1
        terms_per_topic[new_topic] += 1
