#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 15 November 2018

@author: jason
"""

from random import *
import pickle

def compute_doc_term_freqs(term_seqs, fname):
    # try unpickling first
    try:
        with open(fname, 'rb') as file:
            dv, dvv = pickle.load(file)
        no_file = 1
    except FileNotFoundError:
        no_file = 1
    # if no file was found compute frequencies and pickle them
    if no_file:
        # create empty dv (single term freq) and dvv (pair of terms freq) counts
        dv = {}
        dvv = {}
        for d, doc in enumerate(term_seqs):
            print("processing doc number ", d)
            # for each doc create sets of single and pair terms freqs
            terms_indoc = set()
            pairs_indoc = set()
            for s, w1 in enumerate(doc):
                for w2 in doc[s:]:
                    if w1 == w2:
                        terms_indoc.add(w1)
                    else:
                        entry = frozenset([w1, w2])
                        pairs_indoc.add(entry)
            # and update counts accordingly
            for term in terms_indoc:
                if term in dv:
                    dv[term] += 1
                else:
                    dv[term] = 1
            for pair in pairs_indoc:
                if pair in dvv:
                    dvv[pair] += 1
                else:
                    dvv[pair] = 1
        with open(fname, 'wb') as file:
            try:
                pickle.dump((dv, dvv), file, protocol=3)
            except pickle.PicklingError:
                print('unpicklable object')
    return dv, dvv

def gen_term_seqs(num_docs, num_terms, doc_lengths):
    term_seqs = []
    for _ in range(num_docs):
        seq = [randint(0, num_terms) for _ in range(randint(doc_lengths[0], doc_lengths[1]))]
        term_seqs.append(seq)
    return term_seqs

# def main():
#     num_docs = 10
#     num_terms = 15
#     doc_lengths = (10, 20)
#     term_seqs = gen_term_seqs(num_docs, num_terms, doc_lengths)
#
#     print(term_seqs)
#     dv, dvv = compute_doc_term_freqs(term_seqs)
#     print(dv)
#     print(dvv)
#
# if __name__ == '__main__':
#     main()