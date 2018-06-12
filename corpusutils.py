#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 11 June 2018

@author: jason
"""

import numpy as np


def get_seqs_and_counts(corpus, num_topics, num_terms=None):
    """
        Builds the sequences of terms and topics, and the counts of topics in docs,
        terms in topics and term per topic.

    Args:
        corpus:

    Returns:
        term_seqs, topic_seqs, doc_topic_counts, term_topic_counts, terms_per_topic
    """

    if num_terms is None:
        num_terms = 1 + max(corpus.dictionary.keys())

    # Build term_seqs
    term_seqs = []
    for document in corpus:
        term_seq = []
        for term_pair in document:
            term_seq += [term_pair[0]] * int(term_pair[1])
        term_seqs.append(term_seq)
    # Init randomly topic_seqs
    topic_seqs = []
    for di in range(len(term_seqs)):
        topic_seq = np.random.randint(num_topics, size=len(term_seqs[di])).tolist()
        topic_seqs.append(topic_seq)
    # Build doc_topic_counts
    doc_topic_counts = []
    for topic_seq in topic_seqs:
        topic_count = [0] * num_topics
        for topic in topic_seq:
            topic_count[topic] += 1
        doc_topic_counts.append(topic_count)
    # Build term_topic_counts
    term_topic_counts = [None] * num_terms
    for term in range(num_terms):
        term_topic_counts[term] = [0] * num_topics
    for di in range(len(term_seqs)):
        assert len(term_seqs[di]) == len(topic_seqs[di])  # Check if everything is fine
        for term, topic in zip(term_seqs[di], topic_seqs[di]):
            term_topic_counts[term][topic] += 1
    # Sum above across terms to build terms_per_topic
    terms_per_topic = [0] * num_topics
    for topic in range(num_topics):
        for term in range(num_terms):
            terms_per_topic[topic] += term_topic_counts[term][topic]
    return term_seqs, topic_seqs, doc_topic_counts, term_topic_counts, terms_per_topic
