# Fréchet (geometric) mean algorithm on the space P
Python implementation of Fréchet mean algorithm on the space of positive definite matrices with respect to Fisher-Rao metric (geometric mean).

Based on the algorithm that presented in the paper:
"Computing the Karcher mean of symmetric positive definite matrices". Binni & Lannazzo 2013


To install and run tests: 

```
$ conda env create -f environment.yml
$ conda activate frechet_mean
$ pytest [ -o log_cli=true -o log_cli_level=DEBUG ]

```




