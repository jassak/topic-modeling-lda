#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 6 November 2018

@author: jason
"""

from gensim import corpora
import os
from six import iteritems
import string
import re
import enchant
from spacy.lang.en.stop_words import STOP_WORDS as SPACY_STOPS
from utilityclasses import SaveLoad


MY_STOPS = 'for a of the and to in an st is that are with by we as be this on from can ' \
           'which it i all have each or at was not if has these will only our where were ' \
           'such thus given there other but when then also been may its use any over their ' \
           'than same using used more particular new given between first second figure shows ' \
           'shown since along a b c d e f g h i j k l m n o p q r s t u v w x y z  very so after ' \
           'must should they et many through no most from how would do let about almost another ' \
           'some one two three four five six seven eight nine ten zero both therefore does had ' \
           'because did into what i ii iii iv vi vii viii ix xi xii xiii xiv xv xvi xvii xviii ' \
           'fig kl ti illus tic natl fl sci soc cir'

STOPLIST = stoplist = set(MY_STOPS.split() + list(string.punctuation) + list(SPACY_STOPS))

def clean_corpus_texts(corpus_iter):
    corpus_texts = []
    pattern = re.compile('[\W_]+')
    for text in corpus_iter:
        text = pattern.sub(' ', text)
        corpus_texts.append(text)
    return corpus_texts


def make_corpus_dict(corpus_texts):
    texts_tonekized = []
    for text in corpus_texts:
        text = text.lower().split()
        texts_tonekized.append(text)
    dictionary = corpora.Dictionary(texts_tonekized)
    # remove stop words, words that appear only in one doc, non-alpha and non-english words
    stop_ids = [dictionary.token2id[stopword] for stopword in STOPLIST
                if stopword in dictionary.token2id]
    symbol_ids = [tokenid for tokenid in dictionary if not dictionary[tokenid].isalpha()]
    few_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]
    deng = enchant.Dict("en_US")
    noneng_ids = [tokenid for tokenid in dictionary if not deng.check(dictionary[tokenid])]
    dictionary.filter_tokens(
        stop_ids + few_ids + symbol_ids + noneng_ids)  # remove stop words and words that appear only once
    dictionary.compactify()  # remove gaps in id sequence after words that were removed
    return dictionary

def prepare_nips_cropus():
    corpus_iter = NipsIterator()
    corpus_texts = clean_corpus_texts(corpus_iter)
    dictionary = make_corpus_dict(corpus_texts)
    abstr_id = dictionary.token2id['abstract']
    term_seqs = []
    token_texts = []
    for text in corpus_texts:
        text = text.lower().split()
        text = dictionary.doc2idx(text)
        term_seq = []
        abstr_passed = 0
        for w in text:
            if abstr_passed == 1:
                if w != -1:
                    term_seq.append(w)
            if w == abstr_id:
                abstr_passed = 1
        term_seqs.append(term_seq)
        token_texts.append(' '.join([dictionary[w] for w in term_seq]))
    return term_seqs, dictionary, token_texts

class NipsIterator():
    def __iter__(self):
        for f_id in range(13):
            f_suffix = str(f_id).zfill(2)
            dir_name = os.fsencode('data/nipstxt/nips' + f_suffix + '/')
            for f_name in os.listdir(dir_name):
                with open(dir_name + f_name, 'r', encoding="latin-1") as file:
                    yield file.read()

class NipsCorpus(SaveLoad):
    def __init__(self):
        self.term_seqs, self.dict, self.texts = prepare_nips_cropus()
        self.num_docs = self.dict.num_docs
        self.num_terms = len(self.dict)
        self.av_doc_length = int(round((sum([len(seq) for seq in self.term_seqs]) / self.num_docs)))
