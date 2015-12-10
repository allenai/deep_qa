
## Getting the data

The data is not included in the git distribution because it's very
large. To fetch the data and place it in the appropriate directory and
format it for training, run the following commands:

```
mkdir data
cd data
wget http://rtw.ml.cmu.edu/tacl2015_csf/tacl2015-training.txt.gz
gunzip tacl2015-training.txt.gz
./src/scripts/process_training_data.sh
```

This will create four directories under data. Two of them are are
data/predicate and data/query, which are contain training data for the
predicate and query ranking objectives from the paper,
respectively. There is also a variant of each of these suffixed with
"_small" which contains a subsampled version of the original
dataset. Each data directory contains three files:

* entities.txt -- defines the list of entities and entity tuples in the
* category and relation matrices
* words.txt -- defines the list of category and relation predicates
* lf.txt -- training examples

## Training

To train the matrix factorization models with the predicate ranking
objective, run:

```
./src/scripts/train_predicate_ranking.sh
```

For the query ranking objective, run:

```
./src/scripts/train_query_ranking.sh
```

By default, these scripts train on the subsampled predicate/query data
generated above. To use a different data set, change the value of the
`DATA_DIR` variable within them. The output of these scripts will be
directed into the `output/<dataset_name>/` directory. This output
includes a log file that shows the progression of training and a model
file containing the trained models serialized as a Java object.

## Running Models Interactively

You can interactively run queries against a trained model. To start,
run the following command:

```
./src/scripts/run_model.sh
```

This command starts an interactive lisp interpreter. (The command line
is not very usable at the moment.) You can run queries using the
following command:

```
(expression-eval (quote (print-predicate-marginals **QUERY** entity-array)))
```

For example, to get "17th-century painter from France", we can run:

```
(expression-eval (quote (print-predicate-marginals (lambda x (and ((word-cat "17th-century") x) ((word-cat "painter") x) ((word-rel "from") x "/m/0f8l9c"))) entity-array)))
```

Above, "/m/0f8l9c" is the Freebase mid for France. This command prints
out the marginal probability that each entity in entity-array is in
the denotation of the query.

