def time_alias_sampler():
    import timeit
    import numpy as np

    vec_sizes = [1e1, 1e2, 1e3, 1e4]
    times_table = np.zeros((2, len(vec_sizes)))

    for i in range(len(vec_sizes)):
        setup = """\
import numpy as np
from alias_sampler import AliasSampler
p = np.random.rand(
""" + str(int(vec_sizes[i])) + """ )
p = p / sum(p)
p[0] = 1. - sum(p[1:])
alias_sampler = AliasSampler(p, dtype=np.float16)
"""
        times_table[0][i] = timeit.timeit('alias_sampler.generate(1)', number=100000, setup=setup)
        # print(t)

        setup = """\
import numpy as np
p = np.random.rand(
""" + str(int(vec_sizes[i])) + """ )
p = p / sum(p)
p[0] = 1. - sum(p[1:])
"""
        times_table[1][i] = timeit.timeit('np.random.multinomial(1, p, size=1)', number=100000,
                                          setup=setup)
        # print(t)

    print(times_table)


def make_corpus():
    from gensim import corpora
    from pprint import pprint

    documents = ["Human machine interface for lab abc computer applications",
                 "A survey of user opinion of computer system response time",
                 "The EPS user interface management system",
                 "System and human system engineering testing of EPS",
                 "Relation of user perceived response time to error measurement",
                 "The generation of random binary unordered trees",
                 "The intersection graph of paths in trees",
                 "Graph minors IV Widths of trees and well quasi ordering",
                 "Graph minors A survey"]
    # remove common words and tokenize
    stoplist = set('for a of the and to in'.split())
    texts = [[word for word in document.lower().split() if word not in stoplist]
             for document in documents]
    # remove words that appear only once
    from collections import defaultdict
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    texts = [[token for token in text if frequency[token] > 1]
             for text in texts]
    corpus_dict = corpora.Dictionary(texts)
    # print(corpus_dict.token2id)
    corpus = [corpus_dict.doc2bow(text) for text in texts]
    # print(corpus)
    return (corpus, corpus_dict)


if __name__ == '__main__':
    from aliassampler import *
    from ldamodel_cgs import *
    from gensim import corpora
    from nipscorpus import *
    from gensim import models
    import itertools
    from utilityclasses import *
    from corpusutils import get_seqs_and_counts
    from useful_datatypes import SparseCounter

    test_count = SparseCounter([1, 3, 5, 3, 4, 6, 6])

    #
    # corpus = NipsCorpus()
    # model = LDAModelCGS(corpus, num_topics=6, num_passes=2)
    # model.save('models/test_model2.pkl')
    #
    # # model = LDAModelCGS.load('models/test_model.pkl')
    #
    # model.print_document_topics(2)
    # model.print_topic_terms(6)
    # model.print_term_topics(5)
    # model.print_topic_documents(6)
