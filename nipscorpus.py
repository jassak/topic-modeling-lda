#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 1 June 2018

@author: jason
"""

from gensim import corpora
import os
from six import iteritems
import string
import enchant

STOPLIST = stoplist = set('for a of the and to in an st'.split() + list(string.punctuation))


def make_nips_dict():
    nips_iter = NipsIterator()
    dictionary = corpora.Dictionary(text.lower().split() for text in nips_iter)
    # remove stop words, words that appear only in one doc, non-alpha and non-english words
    stop_ids = [dictionary.token2id[stopword] for stopword in STOPLIST
                if stopword in dictionary.token2id]
    few_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]
    symbol_ids = [tokenid for tokenid in dictionary if not dictionary[tokenid].isalpha()]
    deng = enchant.Dict("en_US")
    noneng_ids = [tokenid for tokenid in dictionary if not deng.check(dictionary[tokenid])]
    dictionary.filter_tokens(
        stop_ids + few_ids + symbol_ids + noneng_ids)  # remove stop words and words that appear only once
    dictionary.compactify()  # remove gaps in id sequence after words that were removed
    return dictionary


class NipsIterator():
    def __iter__(self):
        for f_id in range(13):
            f_suffix = str(f_id).zfill(2)
            dir_name = os.fsencode('data/nipstxt/nips' + f_suffix + '/')
            for f_name in os.listdir(dir_name):
                with open(dir_name + f_name, 'r', encoding="latin-1") as file:
                    yield file.read()


class NipsCorpus():
    def __init__(self):
        self.dictionary = make_nips_dict()

    def __iter__(self):
        for f_id in range(13):
            f_suffix = str(f_id).zfill(2)
            dir_name = os.fsencode('data/nipstxt/nips' + f_suffix + '/')
            for f_name in os.listdir(dir_name):
                with open(dir_name + f_name, 'r', encoding="latin-1") as file:
                    yield self.dictionary.doc2bow(file.read().lower().split())
