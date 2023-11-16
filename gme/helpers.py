# SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Delmas Maxime maxime.delmas@idiap.ch
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""helpers functions"""
import itertools
import logging
import random
import sys
from multiprocessing import Pool
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import entropy


def set_seed(seed):
    """
    Set the random seed.
    """
    random.seed(seed)
    np.random.seed(seed)


def read_data(path: str, sep: str, logger: logging.Logger) -> pd.DataFrame:
    """Helper to read a dataframe

    Args:
        path (str): path to the file
        sep (str): separator
        logger (logging.Logger): logger instance

    Returns:
        pd.DataFrame: the dataframe
    """

    logger.info("Read file at %s", path)

    try:
        data = pd.read_csv(path, sep=sep, dtype=object).fillna("")

    except pd.errors.ParserError as except_parsing_error:
        logger.error(
            "file at %s has incorrect format. %s", path, str(except_parsing_error)
        )
        sys.exit(1)

    except FileNotFoundError:
        logger.error("File not found at %s", path)
        sys.exit(1)

    return data


def indexes_to_vector(indexes: List[int], n: int) -> np.ndarray:
    """Convert a vector of indexes to a vector of counts.
    The vector is initialized as a zero-vector of size n.
    The number of times the index i appears in the given list
    of indexes is then used to calculate the value at position i.

    Args:
        indexes (list): the list of indexes
        n (int): the size to give to the vector

    Returns:
        _type_: the vector ()
    """
    if max(indexes) >= n:
        raise ValueError("Maximum index is bigger than size of vector")

    v = np.zeros(n, dtype=np.int32)
    for i in indexes:
        v[i] += 1
    return v


def compute_entropy(indexes_per_doi: list, current_vector: np.array) -> np.array:
    """For each list of indexes provided in indexes_per_doi, compute the entropy that
    would correspond to the addition of these counts to the current entropy vector.
    Recall that each list of indexes correspond the list of involved entities.

    Args:
        indexes_per_doi (list): a list of list of indexes, e-g: [ [1, 4, 60],  [24, 65, 98, 34], ..., [15, 23, 98]]
        current_vector (np.array): the current vector of counts. Each entry correspond to the current number
        of relation involving the ith entity identified by its index.

    Returns:
        np.array: _description_
    """

    n = len(current_vector)
    h_vector = []

    for indexes in indexes_per_doi:
        probs = indexes_to_vector(indexes, n)
        probs += current_vector
        probs = probs / probs.sum()
        e = entropy(probs)
        h_vector.append(np.round(e, decimals=5))

    return h_vector


def compute_all_entropy(l_indexes_per_doi: list, l_occurence_v: list):
    """Giving a list containing a list of vectors for each set of entities,
    along with a list containing the current occurence vector of each set of entities,
    call the entropy function for all of them sequentially.

    Args:
        l_indexes_per_doi (list): [list of list of indexes for each entity type]
        Eg. [[[1,2,3,4], [1,2]], [[4,3,2,1], [2,1]]] where [1,2,3,4] is the set of
        indexes of entities associated with the item 1 in the first variable, and respectively,
        [2,1] is the set of indexes of entities associated with the second variable
        for the second item.
        l_occurence_v (list): [current occurence vector for for each entity type]
        E.g [[0, 0, 1, 2, 0], [0, 2, 1, 0, 0]] The entity at index 2 of the first variable already
        happened 2 times in the current set, while entity at index 0 for instance in the
        second variable do not appears yet.

    Returns:
        _type_: a list of the list of computed entropies
    """
    out = []

    if not len(l_indexes_per_doi) == len(l_occurence_v):
        raise ValueError(
            "Different sizes of the current running occurence vector and the one containing the indexes per items"
        )

    n = len(l_indexes_per_doi)

    for i in range(n):
        next_h = compute_entropy(l_indexes_per_doi[i], l_occurence_v[i])
        out.append(next_h)

    return out


def distribute_entropy_on_workers(
    list_of_indexes_per_item: list, list_of_occurences: list, n_worker: int
) -> list:
    """Parrelise the computation of entropies by ditributing the list of indexes to different workers

    Args:
        list_of_indexes_per_item (list): a list of list of indexes of entities for each variable per DOI
        list_of_occurences (list): the current occurence vector for each variable
        n_worker (int): _description_

    Returns:
        list: list of numpy.arrays containing entropy values.
    """

    if not all(
        [
            len(l_item) == len(list_of_indexes_per_item[0])
            for l_item in list_of_indexes_per_item
        ]
    ):
        raise ValueError("all the list don't have the same size !")
    n = len(list_of_indexes_per_item[0])

    if n <= n_worker or n_worker == 1:
        res = compute_all_entropy(list_of_indexes_per_item, list_of_occurences)
        out = [np.array(x) for x in res]

        return out

    workers_ranges = [(a[0], a[-1]) for a in np.array_split(list(range(n)), n_worker)]

    splited_list_of_indexes_per_item = [
        [l_indexes[r[0] : r[1] + 1] for r in workers_ranges]
        for l_indexes in list_of_indexes_per_item
    ]

    # To send to the worker, we need to split the data, so that each one can working individually in parallel.
    # params in in the form of: [ worker_1, worker_2, ..., worker_n ]. Earch worker in then a list of the parameters
    # that have to be send to compute_all_entropy, so worker_i = [l_indexes_per_doi, l_occurence_v]
    # Then, l_indexes_per_doi is a list, containing the list of indexes for orgamisms and chemical, in the form of:
    #  l_indexes_per_doi = [ [l_o1, l_o2, l_o3, ..., l_on] , [l_c1, l_c2, l_c3, ..., l_cn] ].
    # here n give the nb of example gave to this worked.
    # l_occurence_v in the a list containing the both vector v_org and v_chemical
    # Recall that l_o1 for instance is by itself a list containg all the indexes of the organisms
    # assocaited to this DOI.
    params = list(
        zip(
            (list(zip(*splited_list_of_indexes_per_item))),
            [list_of_occurences for i in range(n_worker)],
        )
    )

    with Pool(processes=n_worker) as pool:
        res = pool.starmap(compute_all_entropy, params)

    out = [list(itertools.chain.from_iterable(r)) for r in zip(*res)]
    out = [np.array(x) for x in out]

    return out
