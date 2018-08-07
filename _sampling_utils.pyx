#!/usr/bin/env cython
# coding: utf-8
# cython: embedsignature=True

# TODO follow gensim: a python wraper using def calling a cython function using cdef
# TODO cythonize do_one_pass
# TODO try various methods of pre-allocation for topic_weights
# TODO check if possible to remove doc_id from args

"""
Created on 5 August 2018

@author: jason
"""

cimport cython
import numpy as np
cimport numpy as np
ctypedef cython.floating DTYPE_t

#def cgs_sample_topics_for_one_doc(doc_id, doc_len, num_topics,
#                                    alpha, beta, w_beta,
#                                    term_seq, topic_seq,
#                                    cur_doc_topic_count, term_topic_counts, terms_per_topic):
#    _cgs_sample_topics_for_one_doc(doc_id, doc_len, num_topics,
#                                    alpha, beta, w_beta,
#                                    term_seq, topic_seq,
#                                    cur_doc_topic_count, term_topic_counts, terms_per_topic)


def cgs_do_one_pass(int num_docs, int num_topics,
                    DTYPE_t[:] alpha, DTYPE_t[:] beta, DTYPE_t w_beta,
                    list term_seqs, list topic_seqs,
                    list doc_topic_counts, list term_topic_counts, list terms_per_topic):

    # cdefs
    cdef int doc_id
    cdef int doc_len

    for doc_id in range(num_docs):
        if doc_id % 10 == 0:
            print(doc_id)
        doc_len = len(term_seqs[doc_id])
        cur_doc_topic_count = doc_topic_counts[doc_id]
        cur_term_seq = term_seqs[doc_id]
        cur_topic_seq = topic_seqs[doc_id]
        cgs_sample_topics_for_one_doc(doc_id, doc_len, num_topics, alpha, beta, w_beta, cur_term_seq, cur_topic_seq,
                                      cur_doc_topic_count, term_topic_counts, terms_per_topic)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cgs_sample_topics_for_one_doc(int doc_id, int doc_len, int num_topics,
                                        DTYPE_t[:] alpha, DTYPE_t[:] beta, DTYPE_t w_beta,
                                        list term_seq, list topic_seq,
                                        list cur_doc_topic_count, list term_topic_counts, list terms_per_topic):

    # cdefs
    cdef int si
    cdef int term_id
    cdef int old_topic
    # TODO try various methods of pre-allocation for topic_weights
#    cdef np.ndarray[DTYPE_t, ndim=1] topic_weights = np.zeros(num_topics, dtype=np.double)
#    cdef double tw
#    cdef double tw_sum

    # Iterate over the positions (words) in the document
    for si in range(doc_len):
        term_id = term_seq[si]
        old_topic = topic_seq[si]
        # logger.debug("sample topics for one doc iteration: position:{0}, term: {1}, old topic: {2}"
        #              .format(si, term_id, old_topic))

        # Remove this topic from all counts
        cur_doc_topic_count[old_topic] -= 1
        term_topic_counts[term_id][old_topic] -= 1
        terms_per_topic[old_topic] -= 1

        # TODO try different methods of building topic_weights
        # IF YOU REMOVE THIS, REMOVE ALLOCS ABOVE ALSO
        # Build a distribution over topics for this term (C-style)
#        tw_sum = 0
#        for ti in range(num_topics):
#            tw = ((term_topic_counts[term_id][ti] + beta[term_id])
#                 / (terms_per_topic[ti] + w_beta)
#                 * (cur_doc_topic_count[ti] + alpha[ti]))
#            topic_weights[ti] = tw
#            tw_sum += tw
#        for ti in range(num_topics):
#            topic_weights[ti] /= tw_sum

        cur_term_topic_count = term_topic_counts[term_id]
        one_beta = beta[term_id]

        # Build a distribution over topics for this term (Python-style)
        topic_weights = [((cur_term_topic_count[ti] + one_beta)
                / (terms_per_topic[ti] + w_beta)
                * (cur_doc_topic_count[ti] + alpha[ti])) for ti in range(num_topics)]
        tw_sum = sum(topic_weights)
        topic_weights = [topic_weights[ti] / tw_sum for ti in range(num_topics)]

        # Sample a topic assignment from this distribution
        new_topic = np.random.choice(num_topics, p=topic_weights)

        # Put that new topic into the counts
        topic_seq[si] = new_topic
        cur_doc_topic_count[new_topic] += 1
        term_topic_counts[term_id][new_topic] += 1
        terms_per_topic[new_topic] += 1


