#!/usr/bin/env cython
# coding: utf-8

"""
Created on 1 November 2018

@author: jason
"""

ctypedef struct StackNode:
    int data
    StackNode * next

ctypedef struct Stack:
    StackNode * root

ctypedef struct CounterNode:
    int label
    int count
    CounterNode * prev
    CounterNode * next

ctypedef struct Counter:
    int nnz
    int nElem
    CounterNode * entry
    CounterNode * head
    CounterNode * tail

ctypedef struct SVNode:
    int key
    int isZero
    double val
    SVNode * next

ctypedef struct SparseVector:
    int nElem
    int nnz
    double norm
    SVNode * entry
    SVNode * head

ctypedef struct SGNode:
        int deg
        int * neighb
        double * weight
        AliasSampler * aliassampler

ctypedef struct SparseGraph:
        int nnodes
        double avdeg
        SGNode * node

ctypedef struct AliasSampler:
    int * aliasTable
    double * probTable

ctypedef struct CorpusData:
    int num_topics
    int num_terms
    int num_docs
    int * doc_len
    int ** cTermSeqs
    int ** cTopicSeqs
    Counter ** cDocTopicCounts
    int ** cTermTopicCounts
    int * cTermsPerTopic

ctypedef struct Priors:
    double * alpha
    double * beta
    double w_beta
