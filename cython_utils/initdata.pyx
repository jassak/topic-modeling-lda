#!/usr/bin/env cython
# coding: utf-8

"""
Created on 1 November 2018

@author: jason
"""

cimport cython
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.math cimport fabs
import numpy as np
cimport numpy as np

# include "rng.pyx"
# include "datastructs.pyx"
# include "initdata.pyx"
# include "stack.pyx"
# include "sparsecounter.pyx"
# include "sparsevector.pyx"
# include "sparsegraph.pyx"
# include "aliassampler.pyx"

def init_seqs_and_counts(num_topics, num_terms, corpus):
    cdef int di, topic, term
    # Build term_seqs[d][s]
    term_seqs = []
    for document in corpus:
        term_seq = []
        for term_pair in document:
            term_seq += [term_pair[0]] * int(term_pair[1])
        term_seqs.append(term_seq)
    # Init randomly topic_seqs[d][s]
    topic_seqs = []
    for di in range(len(term_seqs)):
        # init to a random seq, problem: not sparse
         topic_seq = np.random.randint(num_topics, size=len(term_seqs[di])).tolist()
#         topic_seq = np.random.randint(10, size=len(term_seqs[di])).tolist()
         topic_seqs.append(topic_seq)
    # Build term_topic_counts[w][t]
    term_topic_counts = [None] * num_terms
    for term in range(num_terms):
        term_topic_counts[term] = [0] * num_topics
    for di in range(len(term_seqs)):
        assert len(term_seqs[di]) == len(topic_seqs[di])  # Check if everything is fine
        for term, topic in zip(term_seqs[di], topic_seqs[di]):
            term_topic_counts[term][topic] += 1
    # Sum above across terms to build terms_per_topic[t]
    terms_per_topic = [0] * num_topics
    for topic in range(num_topics):
        for term in range(num_terms):
            terms_per_topic[topic] += term_topic_counts[term][topic]
    return term_seqs, topic_seqs, term_topic_counts, terms_per_topic

@cython.boundscheck(False)
@cython.wraparound(False)
cdef CorpusData * _initCData(int num_topics, int num_terms, int num_docs, int ** cTermSeqs, int ** cTopicSeqs,
    int ** cTermTopicCounts, int *cTermsPerTopic, int * doc_len):
    cdef int d
    cdef CorpusData * cdata
    # cdata malloc
    cdata = <CorpusData *> PyMem_Malloc(sizeof(CorpusData))
    # pack existing vars in CData struct
    cdata.num_topics = num_topics
    cdata.num_terms = num_terms
    cdata.num_docs = num_docs
    cdata.cTermSeqs = cTermSeqs
    cdata.cTopicSeqs = cTopicSeqs
    cdata.cTermTopicCounts = cTermTopicCounts
    cdata.cTermsPerTopic = cTermsPerTopic
    cdata.doc_len = doc_len
    # build cDocTopicCounts[d]
    cdata.cDocTopicCounts = _init_docTopicCounts(num_docs, num_terms, cTopicSeqs, doc_len)
    return cdata

@cython.boundscheck(False)
@cython.wraparound(False)
cdef Counter ** _init_docTopicCounts(int num_docs, int num_terms, int **cTopicSeqs, int * doc_len):
    cdef:
        int d
        Counter ** cDocTopicCounts
    cDocTopicCounts = <Counter **> PyMem_Malloc(num_docs * sizeof(Counter *))
    for d in range(num_docs):
        cDocTopicCounts[d] = newCounter(num_terms)
        countSequence(doc_len[d], num_terms, cTopicSeqs[d], cDocTopicCounts[d])
    return cDocTopicCounts

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
cdef Priors * _init_priors(int num_topics, int num_terms):
    cdef int i
    cdef double * alpha
    cdef double * beta
    cdef double w_beta
    cdef Priors * priors
    # malloc
    priors = <Priors *> PyMem_Malloc(sizeof(Priors))
    alpha = <double *> PyMem_Malloc(num_topics * sizeof(double))
    beta = <double *> PyMem_Malloc(num_terms * sizeof(double))
    # compute priors
    for i in range(num_topics):
        alpha[i] = 1.0 / num_topics
    for i in range(num_terms):
        beta[i] = 1.0 / num_terms
    w_beta = 0.
    for i in range(num_terms):
        w_beta += beta[i]
    priors.alpha = alpha
    priors.beta = beta
    priors.w_beta = w_beta
    return priors

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef SparseGraph * _init_similarity_graph(int num_terms, double lam, list similarity_matrix):
    cdef:
        int i, j
        int ** adjMat
        double ** SMat
        SparseGraph * sg
    # malloc
    adjMat = <int **> PyMem_Malloc(num_terms * sizeof(int *))
    for i in range(num_terms):
        adjMat[i] = <int *> PyMem_Malloc(num_terms * sizeof(int))
    SMat = <double **> PyMem_Malloc(num_terms * sizeof(double *))
    for i in range(num_terms):
        SMat[i] = <double *> PyMem_Malloc(num_terms * sizeof(double))
    # make sure similarity_matrix is stochastic
    for i in range(num_terms):
        sum_row = sum(similarity_matrix[i])
        for j in range(num_terms):
            similarity_matrix[i][j] /= sum_row
    # create S = (1 - lam) * I + lam * similarity_matrix (eq.(8) in Ahmed, Long, Silva, Wang 2017)
    # and its adjacency matrix
    for i in range(num_terms):
        adjMat[i][i] = 1
        SMat[i][i] = 1.0 - lam
        for j in range(i + 1, num_terms):
            if fabs(similarity_matrix[i][j] - 0.0) > 1e-10:
                adjMat[i][j] = 1
                adjMat[j][i] = adjMat[i][j]
                SMat[i][j] = lam * similarity_matrix[i][j]
                SMat[j][i] = SMat[i][j]
            else:
                adjMat[i][j] = 0
                adjMat[j][i] = adjMat[i][j]
                SMat[i][j] = 0.0
                SMat[j][i] = SMat[i][j]
    # create similarity graph base on S
    sg = newSparseGraph(num_terms, adjMat, SMat)
    return sg

def test_sim_mat():
    cdef:
        int i, j, k
        int num_terms = 10
        double lam = 0.5
        double sum
        SparseGraph * simgraph
    # populate randomly
    similarity_matrix = []
    for i in range(num_terms):
        similarity_row = [0] * num_terms
        similarity_matrix.append(similarity_row)
    for i in range(num_terms):
        for j in range(i + 1, num_terms):
            similarity_matrix[i][j] = randInt(0, 2) * randUniform()
            similarity_matrix[j][i] = similarity_matrix[i][j]
    # print similarity_matrix
    print(similarity_matrix)
    # create similarity graph
    simgraph = _init_similarity_graph(num_terms, lam, similarity_matrix)
    # print graph
    printf("node neighbours:\n")
    for i in range(num_terms):
        printf("node %d:", i)
        for k in range(simgraph.node[i].deg):
            printf("\t%d", simgraph.node[i].neighb[k])
        printf("\n")
    printf("node weights:\n");
    for i in range(num_terms):
        printf("node %d:", i)
        sum = 0.0
        for k in range(simgraph.node[i].deg):
            sum += simgraph.node[i].weight[k]
            printf("\t%f", simgraph.node[i].weight[k])
        printf("\nsum = %f\n", sum)
