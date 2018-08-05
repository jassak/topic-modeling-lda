#!/usr/bin/env python
# -*- coding: utf-8 -*-

# TODO write cgs_do_one_pass and then cythonize it

"""
Created on 5 August 2018

@author: jason
"""

import numpy as np

def cgs_do_one_pass(num_docs, num_topics):
    pass

def cgs_sample_topics_for_one_doc(doc_id, doc_len, num_topics, alpha, vec_beta, w_beta, term_seq, topic_seq,
                                  cur_doc_topic_count,
                                  term_topic_counts, terms_per_topic):

    # Iterate over the positions (words) in the document
    for si in range(doc_len):
        term_id = term_seq[si]
        old_topic = topic_seq[si]
        # TODO solve logger problem
        # logger.debug("sample topics for one doc iteration: position:{0}, term: {1}, old topic: {2}"
        #              .format(si, term_id, old_topic))

        # Remove this topic from all counts
        cur_doc_topic_count[old_topic] -= 1
        term_topic_counts[term_id][old_topic] -= 1
        terms_per_topic[old_topic] -= 1

        cur_term_topic_count = term_topic_counts[term_id]
        beta = vec_beta[term_id]

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


