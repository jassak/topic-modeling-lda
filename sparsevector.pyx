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

ctypedef struct NZKeyList:
    int nnz
    int * key

ctypedef struct NZValList:
    int nnz
    double * val

@cython.boundscheck(False)
@cython.wraparound(False)
cdef SparseVector * newSparseVector(int nElem):
    cdef:
        int key
        SparseVector * sv
    sv = <SparseVector *> PyMem_Malloc(sizeof(SparseVector))
    sv.nElem = nElem
    sv.nnz = 0
    sv.entry = <SVNode *> PyMem_Malloc(nElem * sizeof(SVNode))
    sv.head = NULL
    for key in range(nElem):
        initSVNode(key, &sv.entry[key])
    return sv

cdef void initSVNode(int key, SVNode * svn):
    svn.key = key
    svn.isZero = 1
    svn.val = 0.0

cdef void setSVVal(int key, double val, SparseVector * sv):
    if sv.entry[key].isZero:
        sv.entry[key].val = val
        sv.entry[key].isZero = 0
        sv.nnz += 1
        sv.entry[key].next = sv.head
        sv.head = &sv.entry[key]
    else:
        sv.entry[key].val = val

cdef double getSVVal(int key, SparseVector * sv):
    return sv.entry[key].val

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void normalizeSV(SparseVector * sv):
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

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int * getSVnzKeyList(SparseVector * sv):
    cdef:
        int k
        int keynz = sv.head.key
        int * nzkeys
    nzkeys = <int *> PyMem_Malloc(sv.nnz * sizeof(int))
    for k in range(sv.nnz):
        nzkeys[k] = keynz
        if sv.entry[keynz].next != NULL:
            keynz = sv.entry[keynz].next.key
    return nzkeys

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double * getSVnzValList(SparseVector * sv):
    cdef:
        int k
        int keynz
        int * nzkeys
        double * nzvals
    nzvals = <double *> PyMem_Malloc(sv.nnz * sizeof(double))
    nzkeys = getSVnzKeyList(sv)
    for k in range(sv.nnz):
        keynz = nzkeys[k]
        nzvals[k] = sv.entry[keynz].val
    PyMem_Free(nzkeys)
    return nzvals

cdef void freeSparseVector(SparseVector * sv):
    PyMem_Free(sv.entry)
    PyMem_Free(sv)

def test_sparsevector():
    c_test_sparsevector()

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void c_test_sparsevector():
    cdef:
        int k
        int nElem = 1000
        double rr
        SparseVector * sv
        double * nzvals
#    srand(1)
    sv = newSparseVector(nElem)
    for k in range(nElem):
        if not (k % 100):
#            rr = randUniform()
            setSVVal(k, 1, sv)
    normalizeSV(sv)
    nzvals = getSVnzValList(sv)
#    printf("nzvals:\n")
#    for k in range(sv.nnz):
#        printf("%f\t", nzvals[k])
#    printf("\n")
#    printf("norm = %f\n", sv.norm);
    freeSparseVector(sv)


@cython.cdivision(True)
cdef double randUniform() nogil:
    return <double> rand() / RAND_MAX

