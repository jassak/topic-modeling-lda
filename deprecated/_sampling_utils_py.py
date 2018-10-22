#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file contains the python versions of the functions found in _sampling_utils.pyx (cython versions)
I keep this file for python/cython comparisons. To compare performances replace _sampling_utils with
_sampling_utils_py in corresponding model imports.

Created on 5 August 2018

@author: jason
"""

import numpy as np

def cgs_do_one_pass(num_docs, num_topics, alpha, beta, w_beta, term_seqs, topic_seqs, doc_topic_counts,
                    term_topic_counts, terms_per_topic):
    """
    Performs one iteration of Gibbs sampling, across all documents.

    """

    for doc_id in range(num_docs):
        if doc_id % 10 == 0:
            print(doc_id)
        doc_len = len(term_seqs[doc_id])
        cur_doc_topic_count = doc_topic_counts[doc_id]
        cur_term_seq = term_seqs[doc_id]
        cur_topic_seq = topic_seqs[doc_id]

        cgs_sample_topics_for_one_doc(doc_len, num_topics, alpha, beta, w_beta, cur_term_seq, cur_topic_seq,
                                      cur_doc_topic_count, term_topic_counts, terms_per_topic)


def cgs_sample_topics_for_one_doc(doc_len, num_topics, alpha, vec_beta, w_beta, term_seq, topic_seq,
                                  cur_doc_topic_count,
                                  term_topic_counts, terms_per_topic):
    """
    Samples a sequence of topics by performing one pass of collapsed Gibbs sampling
    for one document, according to
    **﻿Griffiths, Steyvers: Finding ﻿scientific topics, PNAS 2004**

    Args:
        doc_id:

    """

    # Iterate over the positions (words) in the document
    for si in range(doc_len):
        term_id = term_seq[si]
        old_topic = topic_seq[si]

        # Remove this topic from all counts
        cur_doc_topic_count[old_topic] -= 1
        term_topic_counts[term_id][old_topic] -= 1
        terms_per_topic[old_topic] -= 1

        # localize some variables
        cur_term_topic_count = term_topic_counts[term_id]
        beta = vec_beta[term_id]

        # Build a distribution over topics for this term
        topic_weights = [((cur_term_topic_count[ti] + beta)
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


