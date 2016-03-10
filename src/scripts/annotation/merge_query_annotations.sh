
QUERY_FILE=~/clone/tacl2015-factorization/data/tacl2015/tacl2015-test-new-lfs-augmented.txt

ANNOTATION_FILE=~/clone/annotation-framework/finished_annotations.tsv

ANNOTATED_QUERY_FILE=~/clone/tacl2015-factorization/data/tacl2015/tacl2015-test-new-lfs-augmented-annotated.txt

./src/scripts/annotation/merge_query_annotations2.py $ANNOTATION_FILE $QUERY_FILE > $ANNOTATED_QUERY_FILE
