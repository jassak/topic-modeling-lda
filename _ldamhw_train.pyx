#!/usr/bin/env cython
# coding: utf-8

"""
Created on 22 October 2018

@author: jason
"""


cimport cython
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.stdlib cimport malloc, free, rand, RAND_MAX, srand
from libc.stdio cimport printf
from libc.math cimport floor, fabs
from libc.time cimport time
from libc.limits cimport INT_MIN
import numpy as np
cimport numpy as np


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

# =============================== Init ============================ #
def init_seqs_and_counts(num_topics, num_terms, corpus):
    cdef int di, topic, term
    # Build term_seqs[d][s]
    term_seqs = []
    for document in corpus:
        term_seq = []
        for term_pair in document:
            term_seq += [term_pair[0]] * int(term_pair[1])
        term_seqs.append(term_seq)
    # Init randomly topic_seqs[d][s]
    topic_seqs = []
    for di in range(len(term_seqs)):
        # init to a random seq, problem: not sparse
         topic_seq = np.random.randint(num_topics, size=len(term_seqs[di])).tolist()
#         topic_seq = np.random.randint(10, size=len(term_seqs[di])).tolist()
         topic_seqs.append(topic_seq)
    # Build term_topic_counts[w][t]
    term_topic_counts = [None] * num_terms
    for term in range(num_terms):
        term_topic_counts[term] = [0] * num_topics
    for di in range(len(term_seqs)):
        assert len(term_seqs[di]) == len(topic_seqs[di])  # Check if everything is fine
        for term, topic in zip(term_seqs[di], topic_seqs[di]):
            term_topic_counts[term][topic] += 1
    # Sum above across terms to build terms_per_topic[t]
    terms_per_topic = [0] * num_topics
    for topic in range(num_topics):
        for term in range(num_terms):
            terms_per_topic[topic] += term_topic_counts[term][topic]
    return term_seqs, topic_seqs, term_topic_counts, terms_per_topic

@cython.boundscheck(False)
@cython.wraparound(False)
cdef CorpusData * _initCData(int num_topics, int num_terms, int num_docs, int ** cTermSeqs, int ** cTopicSeqs,
    int ** cTermTopicCounts, int *cTermsPerTopic, int * doc_len):
    cdef int d
    cdef CorpusData * cdata
    cdef Counter ** cDocTopicCounts
    # cdata malloc
    cdata = <CorpusData *> PyMem_Malloc(sizeof(CorpusData))
    # pack existing vars in CData struct
    cdata.num_topics = num_topics
    cdata.num_terms = num_terms
    cdata.num_docs = num_docs
    cdata.cTermSeqs = cTermSeqs
    cdata.cTopicSeqs = cTopicSeqs
    cdata.cTermTopicCounts = cTermTopicCounts
    cdata.cTermsPerTopic = cTermsPerTopic
    cdata.doc_len = doc_len
    # build cDocTopicCounts[d]
    cDocTopicCounts = <Counter **> PyMem_Malloc(num_docs * sizeof(Counter *))
    for d in range(num_docs):
        cDocTopicCounts[d] = newCounter(num_terms)
        countSequence(doc_len[d], num_terms, cTopicSeqs[d], cDocTopicCounts[d])
    cdata.cDocTopicCounts = cDocTopicCounts
    return cdata

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int ** _cythonize_2dlist(list lists):
    cdef int num_lists = len(lists)
    cdef int list_len
    cdef int i, j
    cdef int ** cLists
    # malloc
    cLists = <int **>PyMem_Malloc(num_lists * sizeof(int *))
    for i in range(num_lists):
        list_len = len(lists[i])
        cLists[i] = <int *>PyMem_Malloc(list_len * sizeof(int))
        # list to C pointer
        for j in range(list_len):
            cLists[i][j] = <int>lists[i][j]
    return cLists

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int * _cythonize_1dlist(list alist):
    cdef int list_len = len(alist)
    cdef int i
    cdef int * cList
    # malloc
    cList = <int *>PyMem_Malloc(list_len * sizeof(int))
    # list to C pointer
    for i in range(list_len):
        cList[i] = <int>alist[i]
    return cList

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef Priors * _init_priors(int num_topics, int num_terms):
    cdef int i
    cdef double * alpha
    cdef double * beta
    cdef double w_beta
    cdef Priors * priors
    # malloc
    priors = <Priors *> PyMem_Malloc(sizeof(Priors))
    alpha = <double *> PyMem_Malloc(num_topics * sizeof(double))
    beta = <double *> PyMem_Malloc(num_terms * sizeof(double))
    # compute priors
    for i in range(num_topics):
        alpha[i] = 1.0 / num_topics
    for i in range(num_terms):
        beta[i] = 1.0 / num_terms
    w_beta = 0.
    for i in range(num_terms):
        w_beta += beta[i]
    priors.alpha = alpha
    priors.beta = beta
    priors.w_beta = w_beta
    return priors
# =============================== End of Init ============================ #

# =============================== Train ============================ #
def train(num_topics, num_passes, corpus):
    cdef int d, t, w
    cdef int num_docs
    cdef int num_terms
    cdef CorpusData * cdata
    cdef int ** cTermSeqs
    cdef int ** cTopicSeqs
    cdef Counter ** cDocTopicCounts
    cdef int ** cTermTopicCounts
    cdef int * cTermsPerTopic
    cdef int * doc_len
    cdef Priors * priors
#    cdef list theta, phi
    cdef double sum

    # init RNG
    srand(1)

    # get num_terms from corpus
    id2word = corpus.dictionary
    num_terms = 1 + max(id2word.keys())
    del id2word

    # init sequences and counts
    term_seqs, topic_seqs, term_topic_counts, terms_per_topic = init_seqs_and_counts(num_topics, num_terms, corpus)
    num_docs = len(term_seqs)

    # init doc_len
    doc_len = <int *>PyMem_Malloc(num_docs * sizeof(int))
    for i in range(num_docs):
        doc_len[i] = len(term_seqs[i])

    # convert lists to C pointers
    cTermSeqs = _cythonize_2dlist(term_seqs)
    cTopicSeqs = _cythonize_2dlist(topic_seqs)
    cTermTopicCounts = _cythonize_2dlist(term_topic_counts)
    cTermsPerTopic = _cythonize_1dlist(terms_per_topic)

    # pack data to C structure cdata
    cdata = _initCData(num_topics, num_terms, num_docs, cTermSeqs, cTopicSeqs, cTermTopicCounts, cTermsPerTopic, doc_len)

    # init priors
    priors = _init_priors(num_topics, num_terms)

    # call cythonized train
    _train(cdata, priors, num_passes)

    # allocate theta, phi
    theta = np.empty(shape=(num_docs, num_topics), dtype=np.float64)
    phi = np.empty(shape=(num_topics, num_terms), dtype=np.float64)

    # computer theta, phi
    for d in range(num_docs):
        sum = 0.0
        for t in range(num_topics):
            theta[d][t] = getCount(t, cdata.cDocTopicCounts[d]) + priors.alpha[t]
            sum += theta[d][t]
        for t in range(num_topics):
            theta[d][t] /= sum
    for t in range(num_topics):
        sum = 0.0
        for w in range(num_terms):
            phi[t][w] = cdata.cTermTopicCounts[w][t] + priors.beta[w]
            sum += phi[t][w]
        for w in range(num_terms):
            phi[t][w] /= sum
    return theta, phi

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _train(CorpusData * cdata, Priors * priors, int num_passes):
    cdef:
        bint accept
        int p, d, s, t, w
        int cur_w
        int old_t, new_t
        int old_dtc, new_dtc
        int num_terms
        int num_topics
        int num_docs
        int ** cTermSeqs
        int ** cTopicSeqs
        Counter ** cDocTopicCounts
        int ** cTermTopicCounts
        int * cTermsPerTopic
        int * doc_len
        Stack ** stale_samples
        double prob_ratio, prob_num, prob_den
        double ** qq
        double * qq_norm
        SparseVector * ppdw

    # unpacking (Maybe passing cdata to subsequent methods is faster, test it)
    num_terms = cdata.num_terms
    num_topics = cdata.num_topics
    num_docs = cdata.num_docs
    doc_len = cdata.doc_len

    # malloc
    stale_samples = <Stack **> PyMem_Malloc(num_terms * sizeof(Stack *))
    for w in range(num_terms):
        stale_samples[w] = newStack()
    qq = <double **> PyMem_Malloc(num_terms * sizeof(double *))
    for w in range(num_terms):
        qq[w] = <double *> PyMem_Malloc(num_topics * sizeof(double))
    qq_norm = <double *> PyMem_Malloc(num_terms * sizeof(double))

    # start monte carlo
    for p in range(num_passes):
        printf("pass: %d\n", p)
        for d in range(num_docs):
            for s in range(doc_len[d]):
                # Get current term and topic
                cur_w = cdata.cTermSeqs[d][s]
                old_t = cdata.cTopicSeqs[d][s]

                # Check if stale samples haven't been generated yet or are exhausted and generate
                # new ones if that's the case.
                if isEmpty(stale_samples[cur_w]):
                    generate_stale_samples(cur_w, cdata, priors, stale_samples, qq, qq_norm)

                # Remove current term from counts
                decrementCounter(old_t, cdata.cDocTopicCounts[d])
                cdata.cTermTopicCounts[cur_w][old_t] -= 1
                cdata.cTermsPerTopic[old_t] -= 1

                # Compute sparse component of conditional topic distribution (p_dw in Li et al. 2014)
                ppdw = compute_sparse_comp(d, cur_w, cdata, priors)

                # Draw from proposal distribution eq.(10) in Li et al. 2014
                new_t = bucket_sampling(ppdw, stale_samples[cur_w], qq_norm[cur_w])

                # Accept new_topic with prob_ratio (M-H step)
                old_dtc = getCount(old_t, cdata.cDocTopicCounts[d])
                new_dtc = getCount(new_t, cdata.cDocTopicCounts[d])
                # prob numerator
                prob_num = (new_dtc + priors.alpha[new_t]) \
                            * (cdata.cTermTopicCounts[cur_w][new_t] + priors.beta[cur_w]) \
                            * (cdata.cTermsPerTopic[old_t] + priors.w_beta) \
                            * ((ppdw.norm * getSVVal(old_t, ppdw)) + (qq_norm[cur_w] * qq[cur_w][old_t]))
                # prob denominator
                prob_den = (old_dtc + priors.alpha[old_t]) \
                            * (cdata.cTermTopicCounts[cur_w][old_t] + priors.beta[cur_w]) \
                            * (cdata.cTermsPerTopic[new_t] + priors.w_beta) \
                            * ((ppdw.norm * getSVVal(new_t, ppdw)) + (qq_norm[cur_w] * qq[cur_w][new_t]))
                # prob ratio
                prob_ratio = prob_num / prob_den
                if prob_ratio >= 1.0:
                    accept = 1
                else:
                    accept = randUniform() < prob_ratio

                # If move is accepted put new topic into seqs and counts
                if accept:
                    cdata.cTopicSeqs[d][s] = new_t
                    incrementCounter(new_t, cdata.cDocTopicCounts[d])
                    cdata.cTermTopicCounts[cur_w][new_t] += 1
                    cdata.cTermsPerTopic[new_t] += 1
                # Else put back old topic
                else:
                    cdata.cTopicSeqs[d][s] = old_t
                    incrementCounter(old_t, cdata.cDocTopicCounts[d])
                    cdata.cTermTopicCounts[cur_w][old_t] += 1
                    cdata.cTermsPerTopic[old_t] += 1

                # dealloc
                freeSparseVector(ppdw)

    # TODO dealloc everything

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void generate_stale_samples(int cur_w, CorpusData * cdata, Priors * priors, Stack ** stale_samples, double ** qq, double * qq_norm):
    cdef:
        int t
    # Compute dense component of conditional topic distribution (q_w in Li et al. 2014)
    qq_norm[cur_w] = 0.0
    for t in range(cdata.num_topics):
        qq[cur_w][t] = priors.alpha[t] * (cdata.cTermTopicCounts[cur_w][t] + priors.beta[cur_w]) \
                       / (cdata.cTermsPerTopic[t] + priors.w_beta)
        qq_norm[cur_w] += qq[cur_w][t]
    for t in range(cdata.num_topics):
        qq[cur_w][t] /= qq_norm[cur_w]
    # Sample num_topics samples from above distribution using the alias method
    genSamplesAlias(cdata.num_topics, cdata.num_topics, qq[cur_w], stale_samples[cur_w])

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef SparseVector * compute_sparse_comp(int d, int cur_w, CorpusData * cdata, Priors * priors):
    cdef:
        int k
        int nztopic
        int dtc
        int nnz = 0
        double val
        SparseVector * ppdw
        int * nzlist
    ppdw = newSparseVector(cdata.num_topics)
    nzlist = getNZList(cdata.cDocTopicCounts[d])
    nnz = cdata.cDocTopicCounts[d].nnz
    for k in range(nnz):
        nztopic = nzlist[k]
        dtc = getCount(nztopic, cdata.cDocTopicCounts[d])
        val = dtc * (cdata.cTermTopicCounts[cur_w][nztopic] + priors.beta[cur_w]) \
                    / (cdata.cTermsPerTopic[nztopic] + priors.w_beta)
        setSVVal(nztopic, val, ppdw)
    normalizeSV(ppdw)
    PyMem_Free(nzlist)
    return ppdw

@cython.cdivision(True)
cdef int bucket_sampling(SparseVector * ppdw, Stack * ssw, double qq_normw):
    cdef:
        int new_t_id, new_t
        int * nzkeys
        double * nzvals
    if randUniform() < ppdw.norm / (ppdw.norm + qq_normw):
        nzkeys = getSVnzKeyList(ppdw)
        nzvals = getSVnzValList(nzkeys, ppdw)
        new_t_id = rand_choice(ppdw.nnz, nzvals)
        new_t = nzkeys[new_t_id]
        PyMem_Free(nzkeys)
        PyMem_Free(nzvals)
        return new_t
    else:
        return pop(ssw)

# =============================== End of Train ============================ #

# ======================= SparseVector ========================= #
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
    PyMem_Free(nzkeys)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int * getSVnzKeyList(SparseVector * sv):
    cdef:
        int k
        int keynz
        int * nzkeys
    if sv.head != NULL:
        keynz = sv.head.key
    else:
        printf("getSVnzKeyList Error: sparse vector is empty!\n")
    nzkeys = <int *> PyMem_Malloc(sv.nnz * sizeof(int))
    for k in range(sv.nnz):
        nzkeys[k] = keynz
        if sv.entry[keynz].next != NULL:
            keynz = sv.entry[keynz].next.key
    return nzkeys

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double * getSVnzValList(int * nzkeys, SparseVector * sv):
    cdef:
        int k
        int keynz
        double * nzvals
    nzvals = <double *> PyMem_Malloc(sv.nnz * sizeof(double))
    for k in range(sv.nnz):
        keynz = nzkeys[k]
        nzvals[k] = sv.entry[keynz].val
    return nzvals

cdef void freeSparseVector(SparseVector * sv):
    PyMem_Free(sv.entry)
    PyMem_Free(sv)
# ===================== End of SparseVector ==================== #

# ======================= SparseCounter ======================== #
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

@cython.boundscheck(False)
@cython.wraparound(False)
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

cdef void initCounterNode(int k, CounterNode * countnode) nogil:
    countnode.label = k
    countnode.count = 0
    countnode.prev = NULL
    countnode.next = NULL

cdef int incrementCounter(int key, Counter * counter) nogil:
    if key >= counter.nElem or key < 0:
        return 1                                                          # ValueError
    if counter.entry[key].count > 0:                                    # case 1: count > 1
        counter.entry[key].count += 1
    elif counter.entry[key].count == 0:
        counter.entry[key].count += 1
        if counter.head == NULL:                                        # case 2: count == 0 and first node in queue
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

cdef int decrementCounter(int key, Counter * counter) nogil:
    if key >= counter.nElem or key < 0:
        return 1                                                               # ValueError
    if counter.entry[key].count > 1:                                           # case 1: count > 1
        counter.entry[key].count -= 1
    elif counter.entry[key].count == 1:
        if counter.head != &counter.entry[key] and counter.tail != &counter.entry[key]:
            counter.entry[key].count -=1                                       # case2: count == 1 and not head or tail
            counter.entry[key].prev.next = counter.entry[key].next
            counter.entry[key].next.prev = counter.entry[key].prev
            counter.entry[key].prev = NULL
            counter.entry[key].next = NULL
        elif counter.head == &counter.entry[key] and counter.tail != counter.head: # case 3: count == 1 and head
            counter.entry[key].count -= 1
            counter.head = counter.entry[key].next
            counter.entry[key].next = NULL
            counter.head.prev = NULL
        elif counter.tail == &counter.entry[key] and counter.tail != counter.head: # case 4: count == 1 and tail
            counter.entry[key].count -= 1
            counter.tail = counter.entry[key].prev
            counter.entry[key].prev = NULL
            counter.tail.next = NULL
        elif counter.head == counter.tail:
            counter.entry[key].count -= 1
            counter.head = NULL
            counter.tail = NULL
        counter.nnz -= 1
    else:                                                                      # case 5: count < 1 ValueError
        printf("decrementCounter ValueError: count < 1 already\n");
        return 2
    return 0

cdef int getCount(int key, Counter * counter) nogil:
    return counter.entry[key].count

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void countSequence(int nseq, int nElem, int * seq, Counter * counter) nogil:
    cdef int i
    for i in range(nseq):
        incrementCounter(seq[i], counter)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int * getNZList(Counter * counter):
    cdef int i, inz
    cdef int * nzlist
    nzlist = <int *> PyMem_Malloc(counter.nnz * sizeof(int))
    inz = counter.head.label
    for i in range(counter.nnz):
        nzlist[i] = inz
        if counter.entry[inz].next != NULL:
            inz = counter.entry[inz].next.label
    return nzlist

#cdef void freeCounter(Counter * counter):
#    PyMem_Free(counter.entry)
#    PyMem_Free(counter)
# ======================= End of SparseCounter ======================== #

# ================================ Alias Sampler =========================== #

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

# ================================ End of Alias Sampler =========================== #

# ================================ Stack =========================== #
ctypedef struct StackNode:
    int data
    StackNode * next

ctypedef struct Stack:
    StackNode * root

cdef Stack * newStack():
    cdef Stack * stack
    stack = <Stack *> PyMem_Malloc(sizeof(Stack *))
    stack.root = NULL
    return stack

cdef StackNode * newStackNode(int data):
    cdef StackNode * sn
    sn = <StackNode *> malloc(sizeof(StackNode *))
    sn.data = data
    return sn

cdef int isEmpty(Stack * stack) nogil:
    return not stack.root

cdef void push(Stack * stack, int data):
    cdef StackNode * sn
    sn = newStackNode(data)
    sn.next = stack.root
    stack.root = sn

cdef int pop(Stack * stack):
    cdef int data
    cdef StackNode * tmp
    if (isEmpty(stack)):
        printf("Error: stack is empty!\n")
        return -INT_MIN
    tmp = stack.root
    data = stack.root.data
    stack.root = stack.root.next
    PyMem_Free(tmp)
    return data


# ================================ End of Stack =========================== #

# =========================== RNG TODO CHANGE RNG!!! ====================== #
@cython.cdivision(True)
cdef double randUniform() nogil:
    return <double> rand() / RAND_MAX

cdef int randInt(int low, int high) nogil:
    return <int> floor((high - low) * randUniform() + low)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef int rand_choice(int n, double * prob) nogil:
    cdef int i
    cdef double r
    cdef double cuml
    r = <double> rand() / RAND_MAX
    cuml = 0.0
    for i in range(n):
        cuml = cuml + prob[i]
        if (r <= cuml):
            return i
# ================================ End of RNG =========================== #