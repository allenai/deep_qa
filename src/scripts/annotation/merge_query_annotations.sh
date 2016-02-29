
QUERY_FILE=~/clone/tacl2015-factorization/data/acl2016-test-augmented.txt

ANNOTATION_FILE=~/clone/annotation-framework2/finished_annotations.tsv

ANNOTATED_QUERY_FILE=~/clone/tacl2015-factorization/data/acl2016-test-augmented-annotated.txt

./src/scripts/annotation/merge_query_annotations2.py $ANNOTATION_FILE $QUERY_FILE > $ANNOTATED_QUERY_FILE
