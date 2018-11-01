#!/usr/bin/env cython
# coding: utf-8

"""
Created on 1 November 2018

@author: jason
"""

cimport cython
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.stdio cimport printf, fflush, stdout

# include "datastructs.pyx"
# include "aliassampler.pyx"
# include "rng.pyx"

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef SparseGraph * newSparseGraph(int nnodes, int ** adjMat, double ** weightsMat):
    cdef:
        int i, j, k
        int deg
        double avdeg
        double sum
        SparseGraph * sg
        int * neighb
        double * weight
    sg = <SparseGraph *> PyMem_Malloc(sizeof(SparseGraph))
    sg.node = <SGNode *> PyMem_Malloc(nnodes * sizeof(SGNode))
    sg.nnodes = nnodes
    avdeg = 0.0
    for i in range(nnodes):
        deg = 0
        for j in range(nnodes):
            deg += adjMat[i][j]
        sg.node[i].deg = deg
        sg.node[i].neighb = <int *> PyMem_Malloc(deg * sizeof(int))
        sg.node[i].weight = <double *> PyMem_Malloc(deg * sizeof(double))
        k = 0
        for j in range(nnodes):
            if (adjMat[i][j] == 1):
                sg.node[i].neighb[k] = j
                sg.node[i].weight[k] = weightsMat[i][j]
                k += 1
        sum = 0.0
        for k in range(sg.node[i].deg):
            sum += sg.node[i].weight[k]
        for k in range(sg.node[i].deg):
            sg.node[i].weight[k] /= sum;
        sg.node[i].aliassampler = <AliasSampler *> PyMem_Malloc(sizeof(AliasSampler))
        sg.node[i].aliassampler.aliasTable = <int *> PyMem_Malloc(deg * sizeof(int))
        sg.node[i].aliassampler.probTable = <double *> PyMem_Malloc(deg * sizeof(double))
        initializeAliasTables(deg, sg.node[i].weight, sg.node[i].aliassampler.probTable, sg.node[i].aliassampler.aliasTable)
        sg.node[i].deg = deg
        avdeg += <double> deg
    sg.avdeg = avdeg / nnodes
    return sg


# =========================== TESTS =============================== #

cdef int ** makeAdjMat(int nnodes):
    cdef:
        int i, j
        int ** adjMat
    adjMat = <int **> PyMem_Malloc(nnodes * sizeof(int *))
    for i in range(nnodes):
        adjMat[i] = <int *> PyMem_Malloc(nnodes * sizeof(int))
    for i in range(nnodes):
        adjMat[i][i] = randInt(0, 2 )
        for j in range(i + 1, nnodes):
            adjMat[i][j] = randInt(0, 2)
            adjMat[j][i] = adjMat[i][j]
    return adjMat

cdef double ** makeWeightsMat(int nnodes, int ** adjMat):
    cdef:
        int i, j
        double ** weightsMat
    weightsMat = <double **> PyMem_Malloc(nnodes * sizeof(double *))
    for i in range(nnodes):
        weightsMat[i] = <double *> PyMem_Malloc(nnodes * sizeof(double))
    for i in range(nnodes):
        for j in range(i, nnodes):
            weightsMat[i][j] = adjMat[i][j] * randUniform()
            weightsMat[j][i] = weightsMat[i][j]

    return weightsMat

cdef void c_test_sparsegraph():
    cdef:
        int i, j, k
        int nnodes = 10
        int ** adjMat
        double ** weightsMat
        SparseGraph * sg
    # init rng
    srand(1)
    # make matrices
    adjMat = makeAdjMat(nnodes)
    weightsMat = makeWeightsMat(nnodes, adjMat)
    # print matrices
    for i in range(nnodes):
        for j in range(nnodes):
            printf("%d ", adjMat[i][j])
        printf("\n")
    for i in range(nnodes):
        for j in range(nnodes):
            printf("%f ", weightsMat[i][j])
        printf("\n")
    # make sparse graph
    sg = newSparseGraph(nnodes, adjMat, weightsMat)
    # print graph
    printf("node neighbours:\n")
    for i in range(nnodes):
        printf("node %d:", i)
        for k in range(sg.node[i].deg):
            printf("\t%d", sg.node[i].neighb[k])
        printf("\n")
    printf("node weights:\n");
    for i in range(nnodes):
        printf("node %d:", i)
        for k in range(sg.node[i].deg):
            printf("\t%f", sg.node[i].weight[k])
        printf("\n")
    # test AliasSampler
    cdef:
        int nsam = 1000000
        int inode = 0
        int * counts
    # malloc
    counts = <int *> PyMem_Malloc(nsam * sizeof(int))
    # run tests
    for k in range(sg.node[inode].deg):
        counts[k] = 0
    for i in range(nsam):
        counts[generateOne(sg.node[inode].deg, sg.node[inode].aliassampler.probTable, sg.node[inode].aliassampler.aliasTable)] += 1
    for k in range(sg.node[inode].deg):
        printf("%f\n", <double> counts[k]/nsam)

def test_sparsegraph():
    c_test_sparsegraph()
