# SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Delmas Maxime maxime.delmas@idiap.ch
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from numpy.testing import assert_array_equal

from gme.helpers import compute_all_entropy
from gme.helpers import compute_entropy
from gme.helpers import indexes_to_vector


def test_indexes_to_vector_returns_correct_output():
    out = indexes_to_vector([1, 1, 2, 2, 4], 10)
    assert_array_equal(out, np.array([0, 2, 2, 0, 1, 0, 0, 0, 0, 0]))


def test_compute_entropy_returns_correct_outputs():
    out = compute_entropy([[1, 2, 3, 4], [1, 2]], [0, 0, 1, 2, 0])
    assert out == [1.27703, 1.05492]


def test_compute_all_entropies_correct_outputs():
    out = compute_all_entropy(
        [[[1, 2, 3, 4], [1, 2]], [[4, 3, 2, 1], [2, 1]]],
        [[0, 0, 1, 2, 0], [0, 2, 1, 0, 0]],
    )
    assert out == [[1.27703, 1.05492], [1.27703, 0.67301]]
