#!/usr/bin/env cython
# coding: utf-8

"""
Created on 22 October 2018

@author: jason
"""


cimport cython
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.stdlib cimport malloc, free, rand, RAND_MAX, srand
from libc.stdio cimport printf
from libc.math cimport floor, fabs
from libc.time cimport time
from libc.limits cimport INT_MIN
import numpy as np
cimport numpy as np

include "cython_utils/rng.pyx"
include "cython_utils/datastructs.pyx"
include "cython_utils/initdata.pyx"
include "cython_utils/stack.pyx"
include "cython_utils/sparsecounter.pyx"
include "cython_utils/sparsevector.pyx"
include "cython_utils/sparsegraph.pyx"
include "cython_utils/aliassampler.pyx"


def train(num_topics, num_passes, corpus):
    cdef int d, t, w
    cdef int num_docs
    cdef int num_terms
    cdef CorpusData * cdata
    cdef int ** cTermSeqs
    cdef int ** cTopicSeqs
    cdef Counter ** cDocTopicCounts
    cdef int ** cTermTopicCounts
    cdef int * cTermsPerTopic
    cdef int * doc_len
    cdef Priors * priors
#    cdef list theta, phi
    cdef double sum

    # init RNG
    srand(1)

    # get num_terms, num_docs from corpus
    num_terms = corpus.num_terms
    num_docs = corpus.num_docs

    # init sequences and counts
    term_seqs, topic_seqs, term_topic_counts, terms_per_topic = init_seqs_and_counts(num_topics, num_terms, corpus)

    # init doc_len
    doc_len = <int *>PyMem_Malloc(num_docs * sizeof(int))
    for i in range(num_docs):
        doc_len[i] = len(term_seqs[i])

    # convert lists to C pointers
    cTermSeqs = _cythonize_2dlist(term_seqs)
    cTopicSeqs = _cythonize_2dlist(topic_seqs)
    cTermTopicCounts = _cythonize_2dlist(term_topic_counts)
    cTermsPerTopic = _cythonize_1dlist(terms_per_topic)

    # pack data to C structure cdata
    cdata = _initCData(num_topics, num_terms, num_docs, cTermSeqs, cTopicSeqs, cTermTopicCounts, cTermsPerTopic, doc_len)

    # init priors
    priors = _init_priors(num_topics, num_terms)

    # call cythonized train
    _train(cdata, priors, num_passes)

    # allocate theta, phi
    theta = np.empty(shape=(num_docs, num_topics), dtype=np.float64)
    phi = np.empty(shape=(num_topics, num_terms), dtype=np.float64)

    # computer theta, phi
    for d in range(num_docs):
        sum = 0.0
        for t in range(num_topics):
            theta[d][t] = getCount(t, cdata.cDocTopicCounts[d]) + priors.alpha[t]
            sum += theta[d][t]
        for t in range(num_topics):
            theta[d][t] /= sum
    for t in range(num_topics):
        sum = 0.0
        for w in range(num_terms):
            phi[t][w] = cdata.cTermTopicCounts[w][t] + priors.beta[w]
            sum += phi[t][w]
        for w in range(num_terms):
            phi[t][w] /= sum
    return theta, phi

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _train(CorpusData * cdata, Priors * priors, int num_passes):
    cdef:
        bint accept
        int p, d, s, t, w
        int cur_w
        int old_t, new_t
        int old_dtc, new_dtc
        int num_terms
        int num_topics
        int num_docs
        int ** cTermSeqs
        int ** cTopicSeqs
        Counter ** cDocTopicCounts
        int ** cTermTopicCounts
        int * cTermsPerTopic
        int * doc_len
        Stack ** stale_samples
        double prob_ratio, prob_num, prob_den
        double ** qq
        double * qq_norm
        SparseVector * ppdw

    # unpacking (Maybe passing cdata to subsequent methods is faster, test it)
    num_terms = cdata.num_terms
    num_topics = cdata.num_topics
    num_docs = cdata.num_docs
    doc_len = cdata.doc_len

    # malloc
    stale_samples = <Stack **> PyMem_Malloc(num_terms * sizeof(Stack *))
    for w in range(num_terms):
        stale_samples[w] = newStack()
    qq = <double **> PyMem_Malloc(num_terms * sizeof(double *))
    for w in range(num_terms):
        qq[w] = <double *> PyMem_Malloc(num_topics * sizeof(double))
    qq_norm = <double *> PyMem_Malloc(num_terms * sizeof(double))

    # start monte carlo
    for p in range(num_passes):
        printf("pass: %d\n", p)
        for d in range(num_docs):
            for s in range(doc_len[d]):
                # Get current term and topic
                cur_w = cdata.cTermSeqs[d][s]
                old_t = cdata.cTopicSeqs[d][s]

                # Check if stale samples haven't been generated yet or are exhausted and generate
                # new ones if that's the case.
                if isEmpty(stale_samples[cur_w]):
                    generate_stale_samples(cur_w, cdata, priors, stale_samples, qq, qq_norm)

                # Remove current term from counts
                decrementCounter(old_t, cdata.cDocTopicCounts[d])
                cdata.cTermTopicCounts[cur_w][old_t] -= 1
                cdata.cTermsPerTopic[old_t] -= 1

                # Compute sparse component of conditional topic distribution (p_dw in Li et al. 2014)
                ppdw = compute_sparse_comp(d, cur_w, cdata, priors)

                # Draw from proposal distribution eq.(10) in Li et al. 2014
                new_t = bucket_sampling(ppdw, stale_samples[cur_w], qq_norm[cur_w])

                # Accept new_topic with prob_ratio (M-H step)
                old_dtc = getCount(old_t, cdata.cDocTopicCounts[d])
                new_dtc = getCount(new_t, cdata.cDocTopicCounts[d])
                # prob numerator
                prob_num = (new_dtc + priors.alpha[new_t]) \
                            * (cdata.cTermTopicCounts[cur_w][new_t] + priors.beta[cur_w]) \
                            * (cdata.cTermsPerTopic[old_t] + priors.w_beta) \
                            * ((ppdw.norm * getSVVal(old_t, ppdw)) + (qq_norm[cur_w] * qq[cur_w][old_t]))
                # prob denominator
                prob_den = (old_dtc + priors.alpha[old_t]) \
                            * (cdata.cTermTopicCounts[cur_w][old_t] + priors.beta[cur_w]) \
                            * (cdata.cTermsPerTopic[new_t] + priors.w_beta) \
                            * ((ppdw.norm * getSVVal(new_t, ppdw)) + (qq_norm[cur_w] * qq[cur_w][new_t]))
                # prob ratio
                prob_ratio = prob_num / prob_den
                if prob_ratio >= 1.0:
                    accept = 1
                else:
                    accept = randUniform() < prob_ratio

                # If move is accepted put new topic into seqs and counts
                if accept:
                    cdata.cTopicSeqs[d][s] = new_t
                    incrementCounter(new_t, cdata.cDocTopicCounts[d])
                    cdata.cTermTopicCounts[cur_w][new_t] += 1
                    cdata.cTermsPerTopic[new_t] += 1
                # Else put back old topic
                else:
                    cdata.cTopicSeqs[d][s] = old_t
                    incrementCounter(old_t, cdata.cDocTopicCounts[d])
                    cdata.cTermTopicCounts[cur_w][old_t] += 1
                    cdata.cTermsPerTopic[old_t] += 1

                # dealloc
                freeSparseVector(ppdw)

    # TODO dealloc everything

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void generate_stale_samples(int cur_w, CorpusData * cdata, Priors * priors, Stack ** stale_samples, double ** qq, double * qq_norm):
    cdef:
        int t
    # Compute dense component of conditional topic distribution (q_w in Li et al. 2014)
    qq_norm[cur_w] = 0.0
    for t in range(cdata.num_topics):
        qq[cur_w][t] = priors.alpha[t] * (cdata.cTermTopicCounts[cur_w][t] + priors.beta[cur_w]) \
                       / (cdata.cTermsPerTopic[t] + priors.w_beta)
        qq_norm[cur_w] += qq[cur_w][t]
    for t in range(cdata.num_topics):
        qq[cur_w][t] /= qq_norm[cur_w]
    # Sample num_topics samples from above distribution using the alias method
    genSamplesAlias(cdata.num_topics, cdata.num_topics, qq[cur_w], stale_samples[cur_w])

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef SparseVector * compute_sparse_comp(int d, int cur_w, CorpusData * cdata, Priors * priors):
    cdef:
        int k
        int nztopic
        int dtc
        int nnz = 0
        double val
        SparseVector * ppdw
        int * nzlist
    ppdw = newSparseVector(cdata.num_topics)
    nzlist = getNZList(cdata.cDocTopicCounts[d])
    nnz = cdata.cDocTopicCounts[d].nnz
    for k in range(nnz):
        nztopic = nzlist[k]
        dtc = getCount(nztopic, cdata.cDocTopicCounts[d])
        val = dtc * (cdata.cTermTopicCounts[cur_w][nztopic] + priors.beta[cur_w]) \
                    / (cdata.cTermsPerTopic[nztopic] + priors.w_beta)
        setSVVal(nztopic, val, ppdw)
    normalizeSV(ppdw)
    PyMem_Free(nzlist)
    return ppdw

@cython.cdivision(True)
cdef int bucket_sampling(SparseVector * ppdw, Stack * ssw, double qq_normw):
    cdef:
        int new_t_id, new_t
        int * nzkeys
        double * nzvals
    if randUniform() < ppdw.norm / (ppdw.norm + qq_normw):
        nzkeys = getSVnzKeyList(ppdw)
        nzvals = getSVnzValList(nzkeys, ppdw)
        new_t_id = rand_choice(ppdw.nnz, nzvals)
        new_t = nzkeys[new_t_id]
        PyMem_Free(nzkeys)
        PyMem_Free(nzvals)
        return new_t
    else:
        return pop(ssw)
