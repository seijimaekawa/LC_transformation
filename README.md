# LC Transformation
This repository includes source files of "GNN Transformation Framework for Improving Efficiency and Scalability":
## Source codes
+ run.ipynb : A Jupyter notebook for running all procedures from precomputation to model training.
+ process_spmm.py : A source file for precomputation of feature aggregation.
+ model.py : A source file for LC version GNNs and baselines.
+ utils.py : Utilities used in `process_spmm.py` and `model.py`. 

## Experimental details for reproducibility
+ parameters.pdf : Detailed information about hyperparameter search space and the best parameter sets to reproduce experimental results in the paper.
+ requirement.txt : A list of packages required for running our codes.

# Supported Models
## Our transformed GNNs
+ GCN_LC
+ JKNet_LC
+ GPRGNN_LC

## Comparisons
+ Non-scalable GNN
  + GCN ([paper](https://arxiv.org/abs/1609.02907), [code](https://github.com/tkipf/pygcn))
  + JKNet ([paper](https://arxiv.org/abs/1806.03536))
  + GPRGNN ([paper](https://openreview.net/forum?id=n6jl7fLxrP), [code](https://github.com/jianhao2016/GPRGNN))
+ Precomputation-based GNN
  + SGC ([paper](https://arxiv.org/abs/1902.07153), [code](https://github.com/Tiiiger/SGC))
  + FSGNN ([paper](https://arxiv.org/abs/2105.07634), [code](https://github.com/sunilkmaurya/FSGNN))
+ Sampling-based GNN
  + shaDow-GNN ([paper](https://arxiv.org/abs/2201.07858), [code](https://github.com/facebookresearch/shaDow_GNN))

# Datasets
  | Dataset   | Nodes | Edges | Features | Classes |
  | ------------------------------------------------------- | ------- | ------- | ------- | ------- |
  | [Flickr](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html) | 89,250   | 899,756 | 500 | 7 |
  | [Reddit](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html)        | 232,965  | 11,606,919 | 602 | 41 |
  | [ogbn-arxiv](https://ogb.stanford.edu/docs/nodeprop/)| 169,343 | 1,166,243 | 128 | 40 |
  | [ogbn-papers100M](https://ogb.stanford.edu/docs/nodeprop/) | 111,059,956 | 1,615,685,872 | 128 | 172 |

# Instruction 
To run our codes, please open [run.ipynb](https://github.com/seijimaekawa/LC_transformation/blob/master/run.ipynb) and execute cells from top to bottom. 
