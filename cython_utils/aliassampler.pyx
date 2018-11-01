#!/usr/bin/env cython
# coding: utf-8

"""
Created on 24 October 2018

@author: jason
"""

cimport cython
from libc.stdlib cimport malloc, free, rand, RAND_MAX, srand
from libc.math cimport floor, fabs
from libc.time cimport time
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.stdio cimport printf

# include "stack.pyx"

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void initializeAliasTables(int k, double * weights, double * probTable, int * aliasTable):
    cdef:
        int i, s, l
        Stack * small
        Stack * large
        double * probScaled
    # malloc
    probScaled = <double *> PyMem_Malloc(k * sizeof(double))
    small = newStack()
    large = newStack()
    # rescale probabilities
    for i in range(k):
        probScaled[i] = <double> k * weights[i]
    # divide scaled probs to small and large
    for i in range(k):
        if 1.0 - probScaled[i] > 1e-10:
            push(small, i)
        else:
            push(large, i)
    # prob reallocation
    while not isEmpty(small) and not isEmpty(large):
        s = pop(small)
        l = pop(large)
        probTable[s] = probScaled[s]
        aliasTable[s] = l
        probScaled[l] = probScaled[s] + probScaled[l] - 1.0
        if 1.0 - probScaled[l] > 1e-10:
            push(small, l)
        else:
            push(large, l)
    # finally, empty small and large
    while not isEmpty(small):
        s = pop(small)
        probTable[s] = 1.0
    while not isEmpty(large):
        l = pop(large)
        probTable[l] = 1.0
    # dealloc
    PyMem_Free(probScaled)
    PyMem_Free(small)
    PyMem_Free(large)

cdef int generateOne(int k, double * probTable, int * aliasTable) nogil:
    cdef:
        int ri
        double rr
    ri = randInt(0, k)
    if fabs( probTable[ri] - 1.0 ) < 1e-10:
        return ri
    else:
        rr = randUniform()
        if rr <= probTable[ri]:
            return ri
        else:
            return aliasTable[ri]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void generateMany(int n, int k, double * probTable, int * aliasTable, Stack * samples):
    cdef int i, s
    for i in range(n):
        s = generateOne(k, probTable, aliasTable)
        push(samples, s)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void genSamplesAlias(int n, int k, double * weights, Stack * samples):
    cdef:
        int i
        int * aliasTable
        double * probTable
    # malloc
    aliasTable = <int *> PyMem_Malloc(k * sizeof(int))
    probTable = <double *> PyMem_Malloc(k * sizeof(double))
    # init tables
    initializeAliasTables(k, weights, probTable, aliasTable)
    # gen samples
    generateMany(n, k, probTable, aliasTable, samples)
    # dealloc
    PyMem_Free(aliasTable)
    PyMem_Free(probTable)


# ========================= Tests ===========================#
# def test_aliasTable():
#     c_test_aliasTable()
#
# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True)
# cdef void c_test_aliasTable():
#     cdef:
#         int i, pp, k = 1000, n = 1000000
#         double w_norm = 0.0
#         int * counts
#         StackNode * samples
#         double * weights
#     # malloc
#     counts = <int *> PyMem_Malloc(k * sizeof(int))
#     weights = <double *> PyMem_Malloc(k * sizeof(double))
#     # init rand
#     srand(time(NULL))
#     # init variables
#     for i in range(k):
#         counts[i] = 0
#         weights[i] = randUniform()
#         w_norm += weights[i]
#     for i in range(k):
#         weights[i] /= w_norm
#     # gen samples
#     genSamplesAlias(n, k, weights, &samples)
#     # count samples
#     for i in range(n):
#         pp = pop(&samples)
#         counts[pp] += 1
#     # print results
# #    for i in range(20):
# #        printf("weights[%i] = %f | freq[%i] = %f\n", i, weights[i], i, <double> counts[i] / n)
#
# def test_stack():
#     cdef int pp
#     cdef StackNode * stack
#     stack = NULL
#
#     srand(time(NULL))
#
#     push(&stack, randInt(0, 99))
#     push(&stack, randInt(0, 99))
#     push(&stack, randInt(0, 99))
#     push(&stack, randInt(0, 99))
#     printf("is empty? %d\n", isEmpty(stack))
#     pp = pop(&stack)
#     printf("pop: %d\n", pp)
#     pp = pop(&stack)
#     printf("pop: %d\n", pp)
#     pp = pop(&stack)
#     printf("pop: %d\n", pp)
#     pp = pop(&stack)
#     printf("pop: %d\n", pp)
# #    pp = pop(&stack)
#     printf("is empty? %d\n", isEmpty(stack))
# ================================ End of Tests =========================== #
