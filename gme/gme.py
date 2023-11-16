# SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Delmas Maxime maxime.delmas@idiap.ch
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""GME sampler"""
import gc
import logging
import os
import random
import sys

import numpy as np
import pandas as pd

from gme.helpers import distribute_entropy_on_workers
from gme.helpers import indexes_to_vector


class GreedyMaximumEntropySampler:
    """The GME class"""

    def __init__(self, selector: str, binarised: bool, **kwargs):
        self.selector = selector
        self.binarised = binarised
        self.logger = kwargs.pop("logger", self.get_std_logger())
        self.n_workers = kwargs.pop("n_workers", 1)

    def get_std_logger(self):
        """Create a default logger

        Returns:
            logging.logger: a default logger
        """

        # set loggers
        log_path = os.path.join(
            "./sampling_GME_" + ("_freq" if not self.binarised else "") + ".log"
        )
        open(log_path, "w", encoding="utf-8").close()
        handlers = [
            logging.FileHandler(filename=log_path),
            logging.StreamHandler(stream=sys.stdout),
        ]
        logging.basicConfig(
            handlers=handlers,
            format="%(asctime)s %(levelname)-8s %(message)s",
            level=logging.DEBUG,
        )
        logger = logging.getLogger("lotus-dataset-creator")

        return logger

    def sample(
        self,
        data: pd.DataFrame,
        N: int,
        item_column: str,
        on_columns: list,
        approx: int = 0,
    ):
        """Create a sample.
        The sampled elements are individual instance from 'item_column'.
        Each item will be select to maximise the entropy its related entites in the 'on_columns' attributes.

        Args:
            N (int): number of item from 'item_column' to sample
            item_column (str): individual column
            on_columns (list): list of columns name for the attributes to maximise entropy on.
            approx (int): Instead of computing all the emtropy values, approximate the best candidate by
              only taking the best one over a random sample of n items. If 0, no approximation.
        """

        def preprocess(data, on_columns: list):
            """Preprocess the table to attribute a unique index to each entities in the 'on_columns' attributes.

            Args:
                data (pd.DataFrame): the target dataFrame, containing the 'on_columns' attributes.
                on_columns (list): list of columns name for the attributes to maximise entropy on.

            Returns:
                processed_data (pd.DataFrame): the processed dataframe with new columns added "X_indexes", indicating
                the index of each entity in its corresponding set.
            """

            def join_indexes(data: pd.DataFrame, column: str) -> pd.DataFrame:
                """Determine the indexes for each distinct elements in a column.

                Args:
                    data (pd.DataFrame): the dataframe
                    column (str): the targeted column, e.g: organism_wikidata or structure_wikidata
                    in the working example.

                Returns:
                    pd.DataFrame: the dataframe with a new column adding a unique index for each distinct entity
                    in the column.
                """
                self.logger.info("Preprocessing column %s", column)
                list_of_entities = list(set(data[column]))
                entities_to_join = pd.DataFrame(
                    {
                        column: list_of_entities,
                        column + "_indexes": range(len(list_of_entities)),
                    }
                )
                return data.merge(entities_to_join, how="left", on=column), len(
                    list_of_entities
                )

            n_items_per_columns = []

            for column in on_columns:
                data, n = join_indexes(data, column)
                n_items_per_columns.append(n)

            return data, n_items_per_columns

        def selector(s_type: str, next_h_values: list, n_items_per_columns) -> int:
            """The selector for the next article to add to the dataset.

            Args:
                s_type (str): the seldctor type
                next_h_values (list): list of np.array of new entropies on columns attributes
                n_items_per_columns (list): list of the cardinality (number of distinct items) per
                columns attributes

            Raises:
                ValueError: an error

            Returns:
                int: the index of the new DOI that maximise the chosen selection criteria.
            """
            agg_type = ["sum", "dutopia"]

            if s_type not in agg_type:
                self.logger.error(
                    "Invalid aggregator type. Expected one of: %s", ",".join(agg_type)
                )
                raise ValueError("Invalid aggregator type")

            if s_type == "sum":
                s = sum(next_h_values)
                return s.argmax()

            if s_type == "dutopia":
                # Dist in the norm of the diff
                coord = np.array(next_h_values)

                diff = coord.T - np.array(
                    [np.log(nb_items) for nb_items in n_items_per_columns]
                )
                d_utopia = np.linalg.norm(diff, axis=1)

                return d_utopia.argmin()

        # Copy of working data
        working_data = data.copy()
        # preprocess the data
        working_data, n_items_per_columns = preprocess(working_data, on_columns)
        n_cols = len(on_columns)

        # Only consider the set of elements if binarised, not the freq
        list_of_indexes_per_item = []
        if self.binarised:
            for column in on_columns:
                list_of_indexes_per_item.append(
                    working_data.groupby(item_column)[column + "_indexes"]
                    .apply(set)
                    .apply(list)
                )
        else:
            for column in on_columns:
                list_of_indexes_per_item.append(
                    working_data.groupby(item_column)[column + "_indexes"].apply(list)
                )

        if not all(
            [
                list(c.index) == list(list_of_indexes_per_item[0].index)
                for c in list_of_indexes_per_item
            ]
        ):
            raise ValueError("Indexes per %s are not in the same order" % (item_column))

        # Extract the list of item. It has been assert that it is in the same order for all the 'on_columns' attributes
        list_of_items = list_of_indexes_per_item[0].index.to_list()

        if N > len(list_of_items):
            self.logger.warning(
                "N=%d greater than the total number of available DOI (%d). Decreasing N to the max size.",
                N,
                len(list_of_items),
            )
            N = len(list_of_items)

        # Transform this in a list
        list_of_indexes_per_item = [
            indexes_per_item.tolist() for indexes_per_item in list_of_indexes_per_item
        ]
        list_of_occurences = [
            np.zeros(n_items, dtype=np.int32) for n_items in n_items_per_columns
        ]

        # init
        h_values = [0 for i in range(len(on_columns))]
        sampled_items = []
        l_iter_h = [[] for i in range(len(on_columns))]

        # loop until N examples have been selected
        while len(sampled_items) < N:
            # Check that the list always have the same
            t = [len(l_items) for l_items in list_of_indexes_per_item]
            t.append(len(list_of_items))
            if not len(set(t)) == 1:
                raise ValueError(
                    "Different number of items between individual items between columns."
                )

            if approx < len(list_of_items) and approx > 0:
                # If an approx sample size was provided, only computed Entropy values on a random sample
                sample_indexes = random.sample(range(len(list_of_items)), approx)
                working_list_of_indexes_per_item = [
                    [list_of_indexes_per_item[i][j] for j in sample_indexes]
                    for i in range(n_cols)
                ]

                # Step 1: Compute all possible entropy
                next_h_values = distribute_entropy_on_workers(
                    working_list_of_indexes_per_item, list_of_occurences, self.n_workers
                )
                gc.collect()

                # Step 2: get the doi maximazing H according to the defined selector
                working_selected = selector(
                    self.selector, next_h_values, n_items_per_columns
                )
                selected = sample_indexes[working_selected]
                item = list_of_items.pop(selected)

                # Extarct corresponding H value and compare to previous. Log if needed (On sample !)
                new_h_values = [
                    l_h_values[working_selected] for l_h_values in next_h_values
                ]

            else:
                # Step 1: Compute all possible entropy
                next_h_values = distribute_entropy_on_workers(
                    list_of_indexes_per_item, list_of_occurences, self.n_workers
                )
                gc.collect()

                # Step 2: get the doi maximazing H according to the defined selector
                selected = selector(self.selector, next_h_values, n_items_per_columns)
                item = list_of_items.pop(selected)

                # Extarct corresponding H value and compare to previous. Log if needed
                new_h_values = [l_h_values[selected] for l_h_values in next_h_values]

            line = "| {:<50} | {:>10}".format(
                item,
                "".join(
                    [
                        "H(" + on_columns[i] + ") = " + f"{new_h_values[i]:.2f}" + " | "
                        for i in range(n_cols)
                    ]
                ),
            )
            self.logger.info("%s", line)

            for i in range(n_cols):
                if new_h_values[i] < h_values[i]:
                    self.logger.warning(
                        "Adding %s decreased H(%s): %.2f -> %.2f",
                        item,
                        on_columns[i],
                        h_values[i],
                        new_h_values[i],
                    )

            # Add the sampled item to the list
            sampled_items.append(item)

            # For all columns attribute
            for i in range(n_cols):
                # Add the new H value
                l_iter_h[i].append(new_h_values[i])

                # Extract the vector, add it to the current occurence vector and remove
                #  the corresponding line in the crosstab
                list_of_occurences[i] += indexes_to_vector(
                    list_of_indexes_per_item[i].pop(selected), n_items_per_columns[i]
                )

                # update new H values
                h_values[i] = new_h_values[i]

        df_H = pd.DataFrame(
            dict(zip(([item_column] + on_columns), ([sampled_items] + l_iter_h)))
        )

        return df_H
