#!/usr/bin/env cython
# coding: utf-8

"""
Created on 23 October 2018

@author: jason
"""
# TODO comments

from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.stdio cimport printf

cdef class SparseCounter:
    cdef Counter * _c_counter

    def __cinit__(self, num_elem):
        self._c_counter = newCounter(<int> num_elem)
        if self._c_counter is NULL:
            raise MemoryError()

    def __dealloc__(self):
        if self._c_counter is not NULL:
            freeCounter(self._c_counter)

    cpdef void increment(self, int key):
        if (incrementCounter(<int> key, self._c_counter)):
            raise ValueError()

    cpdef void decrement(self, int key):
        if (decrementCounter(<int> key, self._c_counter)):
            raise ValueError()

    cpdef int get_count(self, int key):
        return getCount(<int> key, self._c_counter)

    def count_seq(self, list seq):
        cdef int i
        cdef int nseq = len(seq)
        cdef int * cseq
        cseq = <int *> PyMem_Malloc(nseq * sizeof(int))
        for i in range(nseq):
            cseq[i] = <int> seq[i]
        countSequence(<int> nseq, <int> self._c_counter.nElem, cseq, self._c_counter)

    cdef void c_count_seq(self, int nseq, int nElem, int * seq):
        countSequence(<int> nseq, <int> nElem, <int *> seq, self._c_counter)

    def get_nzlist(self):
        cdef int i
        cdef list nzlist = [0] * self._c_counter.nnz
        cdef int * c_nzlist
        c_nzlist = getNZList(self._c_counter)
        for i in range(self._c_counter.nnz):
            nzlist[i] = <int> c_nzlist[i]
        return nzlist

    cdef void c_get_nzlist(self, int * c_nzlist):
        c_nzlist = getNZList(self._c_counter)

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

cdef Counter * newCounter(int nElem):
    cdef int k
    cdef Counter * counter
    counter = <Counter *> PyMem_Malloc(sizeof(Counter))
    counter.entry = <CounterNode *> PyMem_Malloc(nElem * sizeof(CounterNode))
    counter.nnz = 0
    counter.nElem = nElem
    counter.head = NULL
    counter.tail = NULL
    for k in range(nElem):
        initCounterNode(k, &counter.entry[k])
    return counter

cdef void initCounterNode(int k, CounterNode * countnode):
    countnode.label = k
    countnode.count = 0
    countnode.prev = NULL
    countnode.next = NULL

cdef int incrementCounter(int key, Counter * counter):
    if (key >= counter.nElem or key < 0):
        return 1                                                          # ValueError
    if (counter.entry[key].count > 0):                                    # case 1: count > 1
        counter.entry[key].count += 1
    elif (counter.entry[key].count == 0):
        counter.entry[key].count += 1
        if (counter.head == NULL):                                        # case 2: count == 0 and first node in queue
            counter.head = &counter.entry[key]
            counter.tail = &counter.entry[key]
        else:                                                             # case 3: count==0 but not first node in queue
            counter.tail.next = &counter.entry[key]
            counter.entry[key].prev = counter.tail
            counter.tail = &counter.entry[key]
        counter.nnz += 1
    else:                                                                 # case 4: count<0 ValueError
        return 2
    return 0

cdef int decrementCounter(int key, Counter * counter):
    if (key >= counter.nElem or key < 0):
        return 1                                                                 # ValueError
    if (counter.entry[key].count > 1):                                           # case 1: count > 1
        counter.entry[key].count -= 1
    elif (counter.entry[key].count == 1):
        if (counter.head != &counter.entry[key] and counter.tail != &counter.entry[key]):
            counter.entry[key].count -=1                                         # case2: count == 1 and not head or tail
            counter.entry[key].prev.next = counter.entry[key].next
            counter.entry[key].next.prev = counter.entry[key].prev
            counter.entry[key].prev = NULL
            counter.entry[key].next = NULL
        elif (counter.head == &counter.entry[key]):                              # case 3: count == 1 and head
            counter.entry[key].count -= 1
            counter.head = counter.entry[key].next
            counter.entry[key].next = NULL
            counter.head.prev = NULL
        elif (counter.tail == &counter.entry[key]):                              # case 4: count == 1 and tail
            counter.entry[key].count -= 1
            counter.tail = counter.entry[key].prev
            counter.entry[key].prev = NULL
            counter.tail.next = NULL
        counter.nnz -= 1
    else:                                                                        # case 5: count < 1 ValueError
        return 2
    return 0

cdef int getCount(int key, Counter * counter):
    return counter.entry[key].count

cdef void countSequence(int nseq, int nElem, int * seq, Counter * counter):
    cdef int i
    for i in range(nseq):
        incrementCounter(seq[i], counter)

cdef int * getNZList(Counter * counter):
    cdef int i, inz
    cdef int * nzlist
    nzlist = <int *> PyMem_Malloc(counter.nnz * sizeof(int))
    inz = counter.head.label
    for i in range(counter.nnz):
        nzlist[i] = inz
        if (counter.entry[inz].next != NULL):
            inz = counter.entry[inz].next.label
    return nzlist

cdef void freeCounter(Counter * counter):
    PyMem_Free(counter.entry)
    PyMem_Free(counter)
