
## Getting the data

The data is not included in the git distribution because it's very
large. To fetch the data and place it in the appropriate directory and
format it for training, run the following commands:

mkdir data
cd data
wget http://rtw.ml.cmu.edu/tacl2015_csf/tacl2015-training.txt.gz
gunzip tacl2015-training.txt.gz
./src/scripts/process_training_data.sh

This will create four directories under data. Two of them are are
data/predicate and data/query, which are contain training data for the
predicate and query ranking objectives from the paper,
respectively. There is also a variant of each of these suffixed with
"_small" which contains a subsampled version of the original
dataset. Each data directory contains three files:

entities.txt -- defines the list of entities and entity tuples in the
category and relation matrices
words.txt -- defines the list of category and relation predicates
lf.txt -- training examples



