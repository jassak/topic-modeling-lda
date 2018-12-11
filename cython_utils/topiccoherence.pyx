#!/usr/bin/env cython
# coding: utf-8

"""
Created on 8 November 2018

@author: jason
"""

cimport cython
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf
from libc.math cimport floor, fabs
from topic_coherence import compute_doc_term_freqs, gen_term_seqs


include "rng.pyx"
include "stack.pyx"
include "datastructs.pyx"


ctypedef struct TermFreqs:
    int num_terms
    int * single
    SparseVector ** pair

cdef TermFreqs * build_c_freqs(int num_docs, int num_terms, list term_seqs, str fname):
    cdef:
        int w, w1, w2
        TermFreqs * term_freqs
        dict dv, dvv
        list temp_pairs = [None] * num_terms
        frozenset pair
        set spair
    dv, dvv = compute_doc_term_freqs(term_seqs, fname)
    # malloc
    term_freqs = <TermFreqs*> malloc(sizeof(TermFreqs))
    term_freqs.single = <int*> malloc(num_terms * sizeof(int))
    term_freqs.pair = <SparseVector**> malloc(num_terms * sizeof(SparseVector*))
    # build single freqs
    for w in range(num_terms):
        term_freqs.single[w] = 0
    for w in dv:
        term_freqs.single[w] = dv[w]
    # build temp pair freqs
    for w in range(num_terms):
        temp_pairs[w] = []
    for pair in dvv:
        spair = set(pair)
        w1 = spair.pop()
        w2 = spair.pop()
        temp_pairs[w1].append((w2, dvv[pair]))
    # build final pair freqs
    for w in range(num_terms):
        term_freqs.pair[w] = newSparseVector(len(temp_pairs[w]))

    print(temp_pairs)

    for w in range(num_terms):
        term_freqs.pair[w] = newSparseVector(10)

    return term_freqs


def test_topcoh():
    cdef:
        int d, w, s
        int num_terms = 15
        int num_docs = 10
        list term_seqs
        TermFreqs * term_freqs

    term_seqs = gen_term_seqs(num_docs, num_terms, (10, 20))

    # print term seqs
    for doc in term_seqs:
        print(doc)

    # compute freqs
    term_freqs = build_c_freqs(num_docs, num_terms, term_seqs, '../data/tmp_freqs.pkl')

    # print some freqs
    for w in range(num_terms):
        print("word " + str(w) + " appears in " + str(term_freqs.single[w]) + " docs")

    for w in range(num_terms):
        print("nElems in vec " + str(term_freqs.pair[w].nElem))