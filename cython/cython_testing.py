#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from aliassampler import AliasSampler as AliasSamplerCy
from scipy import spatial
import timeit

from deprecated.aliassampler import AliasSampler as AliasSamplerPy


def cy_vs_py(num_el=100, num_samples=100):
    probs = np.random.rand(num_el)
    probs = probs / sum(probs)

    # cython part
    setup = """
import numpy as np
from aliassampler import AliasSampler
num_el = 1000
num_samples = 1000
probs = np.random.rand(num_el)
probs = probs / sum(probs)
    """
    time_cy = timeit.timeit(
            'sampler_cy = AliasSampler(probs);\
            samples_cy = sampler_cy.generate(num_samples)',
            setup=setup, number=100
    )
    print(time_cy)


    # python part
    setup = """
import numpy as np
from deprecated.aliassampler import AliasSampler
num_el = 1000
num_samples = 1000
probs = np.random.rand(num_el)
probs = probs / sum(probs)
        """
    time_py = timeit.timeit(
            'sampler_py = AliasSampler(probs);\
            samples_py = sampler_py.generate(num_samples)',
            setup=setup, number=100
    )
    print(time_py)

    print('cy is ', time_py/time_cy, 'times faster than py')

    # sampler_py = AliasSamplerPy(probs)
    # samples_py = sampler_py.generate(num_samples)
    #
    # counts_cy = [0] * num_el
    # for i in range(num_samples):
    #     counts_cy[samples_cy[i]] += 1 / num_samples
    # dist_cy = 1 - spatial.distance.cosine(probs, counts_cy)
    #
    # counts_py = [0] * num_el
    # for i in range(num_samples):
    #     counts_py[samples_py[i]] += 1 / num_samples
    # dist_py = 1 - spatial.distance.cosine(probs, counts_py)
    #
    # print('dist_cy = ', dist_cy)
    # print('dist_py = ', dist_py)





