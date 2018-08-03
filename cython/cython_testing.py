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





