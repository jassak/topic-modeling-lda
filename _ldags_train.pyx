#!/usr/bin/env cython
# coding: utf-8

"""
Created on 29 October 2018

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



def train(int num_topics, int num_passes, corpus, similarity_matrix, double lam):
    cdef:
        int d, t, w
        int num_docs
        int num_terms
        CorpusData * cdata
        int ** cTermSeqs
        int ** cTopicSeqs
        Counter ** cDocTopicCounts
        int ** cTermTopicCounts
        int * cTermsPerTopic
        int * doc_len
        Priors * priors
        SparseGraph * sim_graph
        double sum

    # init RNG
    srand(1)

    # get num_terms from corpus
    id2word = corpus.dictionary
    num_terms = 1 + max(id2word.keys())
    del id2word

    # verify similarity_matrix size
    assert(len(similarity_matrix) == num_terms, "similarity matrix size != num_terms")

    # build similarity graph
    sim_graph = _init_similarity_graph(num_terms, lam, similarity_matrix)

    # init sequences and counts
    term_seqs, topic_seqs, term_topic_counts, terms_per_topic = init_seqs_and_counts(num_topics, num_terms, corpus)
    num_docs = len(term_seqs)

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
    _train(cdata, priors, num_passes, sim_graph)

    # allocate theta, phi
    theta = np.empty(shape=(num_docs, num_topics), dtype=np.float64)
    phi = np.empty(shape=(num_topics, num_terms), dtype=np.float64)

    # compute theta, phi
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
cdef void _train(CorpusData * cdata, Priors * priors, int num_passes, SparseGraph * sim_graph):
    cdef:
        bint accept
        int p, d, s, t, w
        int cur_w
        int old_t, new_t
        int neighb_w
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
        double * qqS_norm
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
    qqS_norm = <double *> PyMem_Malloc(num_terms * sizeof(double))

    # init stale samples
    init_stale_samples(cdata, priors, stale_samples, qq, qq_norm, qqS_norm, sim_graph)
    # compute qqS_norm = sim_graph . qq_norm
    sparseDotProd(sim_graph, qq_norm, qqS_norm)

    # start monte calro
    for p in range(num_passes):
        printf("pass: %d\n", p)
        for d in range(num_docs):
            for s in range(doc_len[d]):
                # get current term and topic
                cur_w = cdata.cTermSeqs[d][s]
                old_t = cdata.cTopicSeqs[d][s]

                # check if stale samples for term_id are exhausted, generate and recompute qqS_norm[cur_w]
                # TODO Probably needs deletion if sample is popped only in bucket_sampling
                if isEmpty(stale_samples[cur_w]):
                    generate_stale_samples(num_topics + sim_graph.node[cur_w].deg - 1, cur_w, cdata, priors, stale_samples,\
                                        qq, qq_norm, qqS_norm, sim_graph)
                    qqS_norm[cur_w] = sparseRowDotProd(cur_w, sim_graph, qq_norm)

                # remove current term from counts
                decrementCounter(old_t, cdata.cDocTopicCounts[d])
                cdata.cTermTopicCounts[cur_w][old_t] -= 1
                cdata.cTermsPerTopic[old_t] -= 1

                # compute sparse component pdw^S and pdw_norm^S
                ppdwS = compute_sparse_comp(d, cur_w, cdata, priors, sim_graph, 5)

                # draw from proposal distribution q(t, w, d)^S with bucket sampling
                # if new_t returns non-negative ppdwS was used
                # if new_t returns -1 qq (stale_samples) has to be used and stale_samples regenerated if needed
                new_t = bucket_sampling(cur_w, ppdwS, stale_samples, qqS_norm[cur_w], sim_graph)
                if new_t == -1:
                    neighb_w = sampleNodeNeighbour(cur_w, sim_graph)
                    if isEmpty(stale_samples[neighb_w]):
                        generate_stale_samples(num_topics + sim_graph.node[neighb_w].deg - 1, neighb_w, cdata,\
                                                priors, stale_samples, qq, qq_norm, qqS_norm, sim_graph)
                        qqS_norm[cur_w] = sparseRowDotProd(cur_w, sim_graph, qq_norm)
                    new_t = pop(stale_samples[neighb_w])

                # draw a neighbour of cur_w as a shortcut to approximate qqS which will be used in next step (see paper 3.3.1)
                neighb_w = sampleNodeNeighbour(cur_w, sim_graph)

                # accept new_topic with prob_ratio (M-H step)
                old_dtc = getCount(old_t, cdata.cDocTopicCounts[d])
                new_dtc = getCount(new_t, cdata.cDocTopicCounts[d])
                # prob numerator (the last term comes from qqS[cur_w] \approx qq[neighb_w])
                prob_num = (new_dtc + priors.alpha[new_t]) \
                            * (cdata.cTermTopicCounts[cur_w][new_t] + priors.beta[cur_w]) \
                            * (cdata.cTermsPerTopic[old_t] + priors.w_beta) \
                            * ((ppdwS.norm * getSVVal(old_t, ppdwS)) + (qq_norm[cur_w] * qq[neighb_w][old_t]))
                # prob denominator (the last term comes from qqS[cur_w] \approx qq[neighb_w])
                prob_den = (old_dtc + priors.alpha[old_t]) \
                            * (cdata.cTermTopicCounts[cur_w][old_t] + priors.beta[cur_w]) \
                            * (cdata.cTermsPerTopic[new_t] + priors.w_beta) \
                            * ((ppdwS.norm * getSVVal(new_t, ppdwS)) + (qq_norm[cur_w] * qq[neighb_w][new_t]))
                # prob ratio
                prob_ratio = prob_num / prob_den
                if prob_ratio >= 1.0:
                    accept = 1
                else:
                    accept = randUniform() < prob_ratio

                # if move is accepted put new topic into seqs and counts
                if accept:
                    cdata.cTopicSeqs[d][s] = new_t
                    incrementCounter(new_t, cdata.cDocTopicCounts[d])
                    cdata.cTermTopicCounts[cur_w][new_t] += 1
                    cdata.cTermsPerTopic[new_t] += 1
                # else put back old topic
                else:
                    cdata.cTopicSeqs[d][s] = old_t
                    incrementCounter(old_t, cdata.cDocTopicCounts[d])
                    cdata.cTermTopicCounts[cur_w][old_t] += 1
                    cdata.cTermsPerTopic[old_t] += 1

                # dealloc
                freeSparseVector(ppdwS)

    # TODO dealloc everything

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void init_stale_samples(CorpusData * cdata, Priors * priors, Stack ** stale_samples, \
                            double ** qq, double * qq_norm, double * qqS_norm, SparseGraph * sim_graph):
    cdef int w
    for w in range(cdata.num_terms):
        generate_stale_samples(cdata.num_topics + sim_graph.node[w].deg - 1, w, cdata, priors, stale_samples,\
                                qq, qq_norm, qqS_norm, sim_graph)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void generate_stale_samples(int num_sam, int cur_w, CorpusData * cdata, Priors * priors, Stack ** stale_samples,\
                            double ** qq, double * qq_norm, double * qqS_norm, SparseGraph * sim_graph):
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
    genSamplesAlias(num_sam, cdata.num_topics, qq[cur_w], stale_samples[cur_w])

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef SparseVector * compute_sparse_comp(int d, int cur_w, CorpusData * cdata, Priors * priors, \
                                        SparseGraph * sim_graph, int num_neighbs):
    cdef:
        int k
        int n, neighb_w
        int nztopic
        int dtc
        int nnz
        double sum
        double val
        SparseVector * ppdwS
        int * nzlist
    ppdwS = newSparseVector(cdata.num_topics)
    nzlist = getNZList(cdata.cDocTopicCounts[d])
    nnz = cdata.cDocTopicCounts[d].nnz
    for k in range(nnz):
        nztopic = nzlist[k]
        dtc = getCount(nztopic, cdata.cDocTopicCounts[d])
        val = dtc / (cdata.cTermsPerTopic[nztopic] + priors.w_beta)
        sum = 0.0
        for n in range(num_neighbs):
            neighb_w = sampleNodeNeighbour(cur_w, sim_graph)
            sum += cdata.cTermTopicCounts[neighb_w][nztopic] + priors.beta[neighb_w]
        val *= sum / num_neighbs
        setSVVal(nztopic, val, ppdwS)
    normalizeSV(ppdwS)
    PyMem_Free(nzlist)
    return ppdwS

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef int bucket_sampling(int cur_w, SparseVector * ppdwS, Stack ** stale_samples, double qqS_normw, SparseGraph * sim_graph):
    cdef:
        int new_t_id, new_t
        int * nzkeys
        double * nzvals
    if randUniform() < ppdwS.norm / (ppdwS.norm + qqS_normw):
        nzkeys = getSVnzKeyList(ppdwS)
        nzvals = getSVnzValList(nzkeys, ppdwS)
        new_t_id = rand_choice(ppdwS.nnz, nzvals)
        new_t = nzkeys[new_t_id]
        PyMem_Free(nzkeys)
        PyMem_Free(nzvals)
        return new_t
    else:
        return -1
