#!/usr/bin/env cython
# coding: utf-8

"""
Created on 25 October 2018

@author: jason
"""

cimport cython
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.stdio cimport printf
from libc.stdlib cimport rand, RAND_MAX, srand, malloc, free

@cython.boundscheck(False)
@cython.wraparound(False)
cdef SparseVector * newSparseVector(int nElem) nogil:
    cdef:
        int key
        SparseVector * sv
    sv = <SparseVector *> malloc(sizeof(SparseVector))
    sv.nElem = nElem
    sv.nnz = 0
    sv.entry = <SVNode *> malloc(nElem * sizeof(SVNode))
    sv.head = NULL
    for key in range(nElem):
        initSVNode(key, &sv.entry[key])
    return sv

cdef void initSVNode(int key, SVNode * svn) nogil:
    svn.key = key
    svn.isZero = 1
    svn.val = 0.0

cdef void setSVVal(int key, double val, SparseVector * sv) nogil:
    if sv.entry[key].isZero:
        sv.entry[key].val = val
        sv.entry[key].isZero = 0
        sv.nnz += 1
        sv.entry[key].next = sv.head
        sv.head = &sv.entry[key]
    else:
        sv.entry[key].val = val

cdef double getSVVal(int key, SparseVector * sv) nogil:
    return sv.entry[key].val

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void normalizeSV(SparseVector * sv) nogil:
    cdef:
        int k
        int keynz
        double norm = 0.0
        int * nzkeys
    nzkeys = getSVnzKeyList(sv)
    for k in range(sv.nnz):
        keynz = nzkeys[k]
        norm += sv.entry[keynz].val
    sv.norm = norm
    for k in range(sv.nnz):
        keynz = nzkeys[k]
        sv.entry[keynz].val /= norm
    free(nzkeys)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int * getSVnzKeyList(SparseVector * sv) nogil:
    cdef:
        int k
        int keynz
        int * nzkeys
    if sv.head != NULL:
        keynz = sv.head.key
    else:
        printf("getSVnzKeyList Error: sparse vector is empty!\n")
    nzkeys = <int *> malloc(sv.nnz * sizeof(int))
    for k in range(sv.nnz):
        nzkeys[k] = keynz
        if sv.entry[keynz].next != NULL:
            keynz = sv.entry[keynz].next.key
    return nzkeys

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double * getSVnzValList(int * nzkeys, SparseVector * sv) nogil:
    cdef:
        int k
        int keynz
        double * nzvals
    nzvals = <double *> malloc(sv.nnz * sizeof(double))
    for k in range(sv.nnz):
        keynz = nzkeys[k]
        nzvals[k] = sv.entry[keynz].val
    return nzvals

cdef void freeSparseVector(SparseVector * sv) nogil:
    free(sv.entry)
    free(sv)

# def test_sparsevector():
#     c_test_sparsevector()
#
# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef void c_test_sparsevector():
#     cdef:
#         int k
#         int nElem = 1000
#         double rr
#         SparseVector * sv
#         double * nzvals
# #    srand(1)
#     sv = newSparseVector(nElem)
#     for k in range(nElem):
#         if not (k % 100):
# #            rr = randUniform()
#             setSVVal(k, 1, sv)
#     normalizeSV(sv)
#     nzvals = getSVnzValList(sv)
# #    printf("nzvals:\n")
# #    for k in range(sv.nnz):
# #        printf("%f\t", nzvals[k])
# #    printf("\n")
# #    printf("norm = %f\n", sv.norm);
#     freeSparseVector(sv)
#
