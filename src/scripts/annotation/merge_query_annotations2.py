#!/usr/bin/python

import json
import sys

annotation_filename = sys.argv[1]
query_filename = sys.argv[2]

annotations = {}
with open(annotation_filename, 'r') as annotation_file:
    query_str = None
    in_query = False
    true_mids = set([])
    false_mids = set([])

    for line in annotation_file:
        if line.startswith("(lambda"):
            query_str = line.strip()
        elif len(line.strip()) == 0:
            if query_str is not None:
                annotations[query_str] = (true_mids, false_mids)

            query_str = None
            true_mids = set([])
            false_mids = set([])
        elif line.startswith("1 ") or line.startswith("0 "):
            parts = line.split(" ")
            annotation = parts[0]
            mid = parts[1]

            if parts[0] == "1":
                true_mids.add(mid)
            else:
                false_mids.add(mid)
                
with open(query_filename, 'r') as query_file:
    for line in query_file:
        sentence_json = json.loads(line)
        queries = sentence_json["queries"]
        for query in queries:
            query_expression = query["queryExpression"]
            # correct_ids = set(query["correctAnswerIds"])
            # incorrect_ids = set(query["incorrectAnswerIds"]) if query.has_key("incorrectAnswerIds") else set([])

            correct_ids = set()
            incorrect_ids = set()

            if not annotations.has_key(query_expression):
                print >> sys.stderr, "missing query:", query_expression
                continue

            annotated_correct = annotations[query_expression][0]
            annotated_incorrect = annotations[query_expression][1]

            correct_ids.update(annotated_correct)
            incorrect_ids.update(annotated_incorrect)

            if not correct_ids.isdisjoint(incorrect_ids):
                print >> sys.stderr, "inconsistent annotation: "
                print >> sys.stderr, "   ", query_expression
                print >> sys.stderr, "   ", correct_ids.intersection(incorrect_ids)

            query["correctAnswerIds"] = list(correct_ids)
            query["incorrectAnswerIds"] = list(incorrect_ids)

        print json.dumps(sentence_json)
