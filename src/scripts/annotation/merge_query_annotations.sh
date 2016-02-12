
QUERY_FILE=~/data/universal_schema/clueweb/lf/test/all_validation_queries.txt

ANNOTATION_FILE=~/data/universal_schema/clueweb/results/100114/annotations2.txt

ANNOTATED_QUERY_FILE=~/data/universal_schema/clueweb/lf/test/all_validation_queries_attested_annotations2.txt

./src/scripts/us/merge_query_annotations2.py $ANNOTATION_FILE $QUERY_FILE > $ANNOTATED_QUERY_FILE
