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


# Instruction 
To run our codes, please open "run.ipynb" and execute cells from top to bottom. 
