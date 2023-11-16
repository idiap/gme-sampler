# SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Delmas Maxime maxime.delmas@idiap.ch
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pandas as pd
from pandas.testing import assert_frame_equal

from gme.gme import GreedyMaximumEntropySampler


def test_sampling_with_dummy_data_correct_outputs():
    # Load dummies data
    data = pd.read_csv("data/test_data.csv", sep="\t", dtype=object).fillna("")
    expected_output = pd.DataFrame(
        {
            "reference_doi": [
                "10.1016/0039-128X(82)90018-6",
                "10.1055/S-2001-11496",
                "10.1016/J.TALANTA.2005.04.043",
                "10.1016/0039-128X(80)90068-9",
                "10.1016/S0305-1978(00)00054-5",
                "10.1016/0166-6851(85)90115-X",
                "10.1007/BF02535649",
                "10.1007/BF02879004",
                "10.1021/JF60213A026",
                "10.1002/(SICI)1099-1573(199902)13:1<78::AID-PTR384>3.0.CO;2-F",
            ],
            "structure_wikidata": [
                2.77259,
                2.89793,
                3.3234,
                3.57149,
                3.7073,
                3.70688,
                3.8136,
                3.89296,
                3.80794,
                3.78291,
            ],
            "organism_wikidata": [
                0.0,
                1.0291,
                1.33408,
                1.5629,
                1.75229,
                1.96591,
                2.09647,
                2.20116,
                2.37203,
                2.4612,
            ],
        }
    )

    # eval outputs
    sampler = GreedyMaximumEntropySampler(selector="dutopia", binarised=False)
    output = sampler.sample(
        data=data,
        N=10,
        item_column="reference_doi",
        on_columns=["structure_wikidata", "organism_wikidata"],
    )

    assert_frame_equal(output, expected_output)
