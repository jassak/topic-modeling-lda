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

# ================================ Alias Sampler =========================== #

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void initializeAliasTables(int k, double * weights, double * probTable, int * aliasTable):
    cdef:
        int i, s, l
        StackNode * small = NULL
        StackNode * large = NULL
        double * probScaled

    # malloc
    probScaled = <double *> PyMem_Malloc(k * sizeof(double))

    # rescale probabilities
    for i in range(k):
        probScaled[i] = <double> k * weights[i]

    # divide scaled probs to small and large
    for i in range(k):
        if 1.0 - probScaled[i] > 1e-10:
            push(&small, i)
        else:
            push(&large, i)

    # take prob from large and put in small
    while not isEmpty(small) and not isEmpty(large):
        s = pop(&small)
        l = pop(&large)
        probTable[s] = probScaled[s]
        aliasTable[s] = l
        probScaled[l] = probScaled[s] + probScaled[l] - 1.0
        if 1.0 - probScaled[l] > 1e-10:
            push(&small, l)
        else:
            push(&large, l)

    # finally, empty small and large
    while not isEmpty(small):
        s = pop(&small)
        probTable[s] = 1.0
    while not isEmpty(large):
        l = pop(&large)
        probTable[l] = 1.0

cdef int generateOne(int k, double * probTable, int * aliasTable):
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
cdef void generateMany(int n, int k, double * probTable, int * aliasTable, int * samples):
    cdef int i
    for i in range(n):
        samples[i] = generateOne(k, probTable, aliasTable)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef int * genSamplesAlias(int n, int k, double * weights):
    cdef:
        int i
        int * aliasTable
        double * probTable
        int * samples

    # malloc
    aliasTable = <int *> PyMem_Malloc(k * sizeof(int))
    probTable = <double *> PyMem_Malloc(k * sizeof(double))
    samples = <int *> PyMem_Malloc(n * sizeof(int))

    # init tables
    initializeAliasTables(k, weights, probTable, aliasTable)

    # gen samples
    generateMany(n, k, probTable, aliasTable, samples)

    # dealloc
    PyMem_Free(aliasTable)
    PyMem_Free(probTable)

    return samples
# ================================ End of Alias Sampler =========================== #
# ================================ Stack =========================== #
ctypedef struct StackNode:
    int data
    StackNode * next

cdef StackNode * newStackNode(int data):
    cdef StackNode * stackNode
    stackNode = <StackNode *> PyMem_Malloc(sizeof(StackNode))
    stackNode.data = data
    stackNode.next = NULL
    return stackNode

cdef bint isEmpty(StackNode * root):
    return not root

cdef void push(StackNode ** root, int data):
    cdef StackNode * stackNode
    stackNode = newStackNode(data)
    stackNode.next = root[0]
    root[0] = stackNode

cdef int pop(StackNode ** root):
    if isEmpty(root[0]):
        printf("pop error: stack empty!\n")
    cdef StackNode * tmp
    cdef int popped
    tmp = root[0]
    root[0] = root[0].next
    popped = tmp.data
    PyMem_Free(tmp)
    return popped
# ================================ End of Stack =========================== #

# ========================= Tests ===========================#
def test_aliasTable():
    c_test_aliasTable()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void c_test_aliasTable():
    cdef:
        int i, k = 1000, n = 1000000
        double w_norm = 0.0
        int * counts
        int * samples
        double * weights
    # malloc
    counts = <int *> PyMem_Malloc(k * sizeof(int))
    weights = <double *> PyMem_Malloc(k * sizeof(double))
    # init rand
    srand(time(NULL))
    # init variables
    for i in range(k):
        counts[i] = 0
        weights[i] = randUniform()
        w_norm += weights[i]
    for i in range(k):
        weights[i] /= w_norm
    # gen samples
    samples = genSamplesAlias(n, k, weights)
    # count samples
    for i in range(n):
        counts[samples[i]] += 1
    # print results
#    for i in range(20):
#        printf("weights[%i] = %f | freq[%i] = %f\n", i, weights[i], i, <double> counts[i] / n)

def test_stack():
    cdef int pp
    cdef StackNode * stack
    stack = NULL

    srand(time(NULL))

    push(&stack, randInt(0, 99))
    push(&stack, randInt(0, 99))
    push(&stack, randInt(0, 99))
    push(&stack, randInt(0, 99))
    printf("is empty? %d\n", isEmpty(stack))
    pp = pop(&stack)
    printf("pop: %d\n", pp)
    pp = pop(&stack)
    printf("pop: %d\n", pp)
    pp = pop(&stack)
    printf("pop: %d\n", pp)
    pp = pop(&stack)
    printf("pop: %d\n", pp)
#    pp = pop(&stack)
    printf("is empty? %d\n", isEmpty(stack))
# ================================ End of Tests =========================== #


# =========================== RNG TODO CHANGE RNG!!! ====================== #
@cython.cdivision(True)
cdef double randUniform():
    cdef double r
    return <double> rand() / RAND_MAX

cdef int randInt(int low, int high):
    return <int> floor((high - low) * randUniform() + low)
# ================================ End of RNG =========================== #
