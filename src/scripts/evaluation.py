#!/usr/bin/python

import sys
import json
import subprocess
import fcntl, os
import time
import re

from query_test_utils import *

cat_query_format = '(expression-eval (quote (print-predicate-marginals %s (get-all-related-entities (list %s)))))'
# cat_query_format = '(expression-eval (quote (print-predicate-marginals %s freq-entities)))'
rel_query_format = '(expression-eval (quote (print-relation-marginals %s entity-tuple-array)))'
# rel_query_format = '(expression-eval (quote (print-relation-marginals %s freq-entity-tuples)))'

def parse_result(lines):
    parsed = []
    for line in lines:
        if re.match("^[0-9].*", line):
            parts = line.split(" ")
            score = float(parts[0])
            mid = parts[1]
            names = entity_names[mid]
            parsed.append((score, mid, names))

    parsed.sort(reverse=True)
    return parsed

def parse_result_relation(lines):
    parsed = []
    for line in lines:
        if re.match("^[0-9].*", line):
            parts = line.split(" ")
            score = float(parts[0])
            mid = parts[2]
            mid2 = parts[3][:-1]
            names = entity_names[mid]
            names2 = entity_names[mid2]

            if score > 0:
                parsed.append((score, (mid,  mid2), (names, names2)))

    parsed.sort(reverse=True)
    return parsed

def send_query(query_string):
    process.stdin.write(query_string + "\n")
    output_text = read_until_prompt()
    return output_text

def read_until_prompt():
    output_text = ''
    while True:
        try:
            read_text = process.stdout.read()

            if DEBUG:
                print >> sys.stderr, read_text

            output_text += read_text
        except IOError:
            pass

        if output_text.endswith('>> '):
            break
        time.sleep(.1)
    return output_text


# Start of the actual script:

with open(query_filename, 'r') as query_file:
    average_precision_sum = 0.0
    weighted_ap_sum = 0.0
    num_queries = 0
    num_no_answer_queries = 0
    num_annotated_true_entities = 0
    num_annotated_false_entities = 0

    point_precision_sum = [0.0] * 11
    for line in query_file:
        sentence_json = json.loads(line)
        queries = sentence_json["queries"]
        for query in queries:
            query_expression = query["queryExpression"]
            mids_in_query = query["midsInQuery"]
            correct_ids = set(query["correctAnswerIds"])
            num_annotated_true_entities += len(correct_ids)
            incorrect_ids = set()
            if query.has_key("incorrectAnswerIds"):
                incorrect_ids = set(query["incorrectAnswerIds"])

            num_annotated_false_entities += len(incorrect_ids)

            is_relation_query = query.has_key("isRelationQuery") and query["isRelationQuery"] == 1

            print ""
            print sentence_json["sentence"]
            print query_expression

            if DEBUG:
                print >> sys.stderr, query_expression

            if is_relation_query:
                complete_query_expression = rel_query_format % (query_expression)
            else:
                complete_query_expression = cat_query_format % (query_expression, " ".join(['"%s"' % x for x in mids_in_query]))
                # complete_query_expression = cat_query_format % (query_expression)
            result = send_query(complete_query_expression)

            lines = result.split("\n")
            if is_relation_query:
                parsed_results = parse_result_relation(lines)
                print "\n".join([str(x[0]) + " " + x[1][0] + " " + x[1][1] + " " 
                                 + "[" + " ".join(['"' + y + '"' for y in x[2][0]]) + "] " 
                                 + "[" + " ".join(['"' + y + '"' for y in x[2][1]]) + "]" 
                                 for x in parsed_results[:pool_depth]])
            else:
                parsed_results = parse_result(lines)
                print "\n".join([("1" if x[1] in correct_ids else ("0" if x[1] in incorrect_ids else "?")) + " " + str(x[0]) + " " + x[1] + " " + " ".join(['"' + y + '"' for y in x[2]])
                                 for x in parsed_results[:pool_depth]])

            if len(correct_ids) == 0:
                num_no_answer_queries += 1

            filtered_parsed_results = [x for x in parsed_results if x[0] > 0.0]
            
            ap = compute_average_precision(filtered_parsed_results[:run_depth], correct_ids)
            print "AP:", ap
            average_precision_sum += ap
            weighted_ap = ap * len(correct_ids)
            print "WAP:", weighted_ap
            weighted_ap_sum += weighted_ap
            num_queries += 1
            
            point_precision = compute_11point_precision(filtered_parsed_results, correct_ids)
            print "11-point precision/recall:", point_precision

            for i in xrange(len(point_precision_sum)):
                point_precision_sum[i] += point_precision[i]

    print "MAP:", average_precision_sum / num_queries
    print "Weighted MAP:", weighted_ap_sum / num_annotated_true_entities
    print "reweighted MAP:", average_precision_sum / (num_queries - num_no_answer_queries)

    average_point_precision = [x / num_queries for x in point_precision_sum]

    print "11-point averaged precision/recall:", average_point_precision
    print "---"
    print "Num queries:", num_queries
    print "annotated true answers:", num_annotated_true_entities
    print "annotated false answers:", num_annotated_false_entities
    print "Queries with no possible answers:", num_no_answer_queries
