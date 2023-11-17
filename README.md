# Greedy Maximum Entropy Sampler

The objective is to extract a sample $S$ of items from an initial set $D$ with an optimized diversity of the attached entities: $S \subset D$, $|S| = l$ and $|D| = L$.

In the initial set $D$, each document (or item) $d$ is attached with a set of $N$-ary relations: $\{r_1, r_2, ..., r_{n_d}\}^{d}$, where $n_d$ in the number of relations attached to $d$. Each relation $r$ is connecting a set of $N$ entities, $r=(e_1, e_2, e_3, e_N)$, each belonging one particular *category* represented by a larger set and denoted by the variables $V_1, V_2, ..., V_{N}$. Sets are considered distincts and each entity belongs to one set: $e \in V_1 \cup V_2 ... \cup V_N$ and $\forall i, j, V_i \cap V_j = \emptyset$.


Also, one entity $e_i$ can appear many times in $d$: $0 \le |\{ r_k: e_i \in r_k\}^{d}| \le n_d$, with $|\{r_k: e_i \in r_k \}^{d}| \text{ the number of }N \text{-ary relations involving }e_i\text{  in }d\text{ .}$

Then, given a subset $S$, the probability that a $N$-ary relation involve the entity $e_i$ is:

$P(e_i) = \frac{\displaystyle\sum_{d \in S} |\{ r_k: e_i \in r_k \}^{d}|}{\displaystyle\sum_{d \in S} n_d}$


It follows that the diversity of organisms or chemicals can be measured with the Shannon's entropy over the probability distributions of elements $V_{1:N}$. Expressed with entropy, the diversity reflects the uncertainty about the entities attached to a relation in an document $d$. The Shannon's entropy in the sample $S$ for the set $V_j$ is $H_S(V_j) = - \displaystyle\sum_{e_i \in V_i} P(e_i) \log P(e_i)$. Adding a new document $\displaystyle d$ to $S$ will update the probability distributions of $V_{1:N}$, and the new observed entropy of $V_j$ for instance will be $H_{\displaystyle S_{+d}}(V_j)$ where $\displaystyle S_{+d} = S \cup \{d\}$. Therefore, to optimize the diversity over $V_{1:N}$, the document $d*$ added to $S$ minimizes the distance to the utopian point $(log |V_1|, log |V_2|, ..., log |V_N|)$ (maximal observable entropies):

$d* = \underset{d}{argmin} || (H_{\displaystyle S_{+d}}(V_1), H_{\displaystyle S_{+d}}(V_2), ..., H_{\displaystyle S_{+d}}(V_N)) - (log |V_1|, log |V_2|, ..., log |V_N|) ||.$

The proposed sampling approach is a simple greedy algorithm that, at each step, selects and adds the new document $d*$ from $D$, maximising the diversity over the different set of entities  $V_{1:N}$. We refer to it as diversity-sampling. As in ecology, diversity is intrinsically related to the number of distinct entities per each set, but, for the purpose of using the samples as training or evaluation sets, the balance of the distribution is prioritized over a higher number of rare entities. This latter behaviours is a natural consequence of using the Shannon's entropy. From this perspective, the method can also been seen as a ranking procedure, and a sample is determined by selecting the first top $n$ ranked items. The selection of an appropriate sample size $l$ is also a critical, but often overlooked factor. By monitoring $H_S(V_{j})$ $\forall j \in 1:N$ during the iterative construction of $S$ (until $l=L$), it is possible to determine the step $l$ at which diversity starts to deteriorate and sampling should be stopped, i.e. when the new added documents provide relationships for already frequently reported entities in $S$.



## Install

Install the gme-sampler
```bash
pip install git+ssh://git@github.com/idiap/gme-sampler
```

## Usage

The provided example considers the motivating scenario of sampling literature references for the LOTUS database. Only a subset of the original database is used in this example.
In this example, each document $d$ is identified by the *reference_doi* columns, e.g: [10.1002/(SICI)1099-1573(199902)13:13.0.CO;2-F](https://onlinelibrary.wiley.com/doi/10.1002/(SICI)1099-1573(199902)13:1%3C75::AID-PTR387%3E3.0.CO;2-F). Each document then reports a list of relations between organisms and chemicals, respectively identified by a Wikidata Organisms entity and a Wikidata structure entity. In this example, we sample to top-5 documents that maximise the diversity among organisms and chemicals from this subset. The dataset looks like this:



| structure_wikidata                        | structure_cid | structure_nameTraditional | organism_wikidata                       | organism_name     | organism_taxonomy_02kingdom | reference_wikidata                       | reference_doi                                 | reference_pubmed_id |
|-------------------------------------------|---------------|---------------------------|-----------------------------------------|-------------------|-----------------------------|------------------------------------------|-----------------------------------------------|---------------------|
| http://www.wikidata.org/entity/Q43656     | 5997          | Cholesterol               | http://www.wikidata.org/entity/Q1146782 | Eryngium foetidum | Archaeplastida              | http://www.wikidata.org/entity/Q34502919 | 10.1002/(SICI)1099-1573(199902)13:13.0.CO;2-F | 10189959            |
| http://www.wikidata.org/entity/Q121802    | 222284        | Beta-Sitosterol           | http://www.wikidata.org/entity/Q1146782 | Eryngium foetidum | Archaeplastida              | http://www.wikidata.org/entity/Q34502919 | 10.1002/(SICI)1099-1573(199902)13:13.0.CO;2-F | 10189959            |
| http://www.wikidata.org/entity/Q104253515 | 5283638       | Clerosterol               | http://www.wikidata.org/entity/Q1146782 | Eryngium foetidum | Archaeplastida              | http://www.wikidata.org/entity/Q34502919 | 10.1002/(SICI)1099-1573(199902)13:13.0.CO;2-F | 10189959            |


- The *item_column* indicates the column containing the documents identifiers while columns containing the variables to maximise the diversity are specified with the *on_columns* argument.

- If *binarised* is set to True, thenwe only consider the distinct set of entities per documents.

- *dutopia* is the main metric use to select the next document are represent the distance to the uptopian point, see [here](#greedy-maximum-entropy-sampler). However, a second metric is provided, *sum*, which simply rank the documents by the sum of the associated entropies.

```python
import pandas as pd
from gme.gme import GreedyMaximumEntropySampler

# Load data
data = pd.read_csv("data/test_data.csv", sep="\t", dtype=object)

# Init sampler
sampler = GreedyMaximumEntropySampler(selector="dutopia", binarised=False)

# Sample
output = sampler.sample(
        data=data,
        N=5,
        item_column="reference_doi",
        on_columns=["structure_wikidata", "organism_wikidata"],
    )
```

The expected output is:

|         reference_doi         | structure_wikidata | organism_wikidata |
|:-----------------------------:|:------------------:|:-----------------:|
|  10.1016/0039-128X(82)90018-6 |       2.77259      |      0.00000      |
|      10.1055/S-2001-11496     |       2.89793      |      1.02910      |
| 10.1016/J.TALANTA.2005.04.043 |       3.32340      |      1.33408      |
|  10.1016/0039-128X(80)90068-9 |       3.57149      |      1.56290      |
| 10.1016/S0305-1978(00)00054-5 |       3.70730      |      1.75229      |

At each step the document which maximised the diversity is sampled and the corresponding increasing entropy values for each variables (here *structure_wikidata* and *organism_wikidata*) are indicated.

On larger datasets, monitoring the entropy values can help estimate a sufficient sample size. See a complete example in our related [article](http://arxiv.org/abs/2311.06364).

## Additional information

The approache scales quadratically with the size of the dataset, which can be impractical for very large datasets. In this context, we implemented an ```--approx``` option. With this option, instead of computing all the emtropy values, the best candidate is approximated by only taking the best one over a random sample of $m$ items. If 0 (default), no approximation is performed.

Considering an intial sample of $N$ items and a used (*approx*) sample size of $k$, we can estimate the probability that the document $d*$ selected (top-1 in the sample of size $k$) is at least in the top-$n$ in the full dataset, with: $1 - [{(N - n) \choose k} / {N \choose k}]$.

For instance, for a dataset of size $N=130,000$, with a sample for *approx* of size $25,000$ (reduced by a factor $5$), we can estimate that there is:
- 88.19 % that it is in the top-10 in the full dataset.
- 98.60 % that it is in the top-20 in the full dataset.
- 99.99 % that it is in the top-50 in the full dataset.
- $\approx 100$ % that it is in the top-100 in the full dataset.
