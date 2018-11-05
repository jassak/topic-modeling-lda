#!/usr/bin/env cython
# coding: utf-8

"""
Created on 1 November 2018

@author: jason
"""

cimport cython
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.limits cimport INT_MIN
from libc.stdio cimport printf

cdef Stack * newStack() nogil:
    cdef Stack * stack
    stack = <Stack *> malloc(sizeof(Stack *))
    stack.root = NULL
    return stack

cdef StackNode * newStackNode(int data) nogil:
    cdef StackNode * sn
    sn = <StackNode *> malloc(sizeof(StackNode *))
    sn.data = data
    return sn

cdef int isEmpty(Stack * stack) nogil:
    return not stack.root

cdef void push(Stack * stack, int data) nogil:
    cdef StackNode * sn
    sn = newStackNode(data)
    sn.next = stack.root
    stack.root = sn

cdef int pop(Stack * stack) nogil:
    cdef int data
    cdef StackNode * tmp
    if (isEmpty(stack)):
        printf("Error: stack is empty!\n")
        return -INT_MIN
    tmp = stack.root
    data = stack.root.data
    stack.root = stack.root.next
    free(tmp)
    return data
