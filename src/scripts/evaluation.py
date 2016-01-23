#!/usr/bin/python

import sys
import json
import subprocess
import fcntl, os
import time
import re

from query_test_utils import *

model_filename = sys.argv[1]
baseline_model_filename = sys.argv[2]
query_filename = sys.argv[3]
all_data_file = sys.argv[4]
data_dir = sys.argv[5]
model_type = sys.argv[6]

cat_query_format = '(expression-eval (quote (print-predicate-marginals %s (get-all-related-entities (list %s)))))'
# cat_query_format = '(expression-eval (quote (print-predicate-marginals %s freq-entities)))'
rel_query_format = '(expression-eval (quote (print-relation-marginals %s entity-tuple-array)))'
# rel_query_format = '(expression-eval (quote (print-relation-marginals %s freq-entity-tuples)))'

pool_depth=100
run_depth=200

entity_file= data_dir + '/entities.txt'
# freq_entity_file= data_dir + '/10k_freq_entities.txt'
word_file= data_dir + '/words.txt'

DEBUG=True

def read_entity_names(midfilename):
    entity_names = {}
    with open(midfilename, 'r') as midfile:
        for line in midfile:
            parts = line.split('\t')
            
            midpart = parts[0]
            namepart = parts[1]
            
            mids = midpart.split(" ")
            names = namepart.strip().split('" "')
            
            for i in xrange(len(mids)):
                if not entity_names.has_key(mids[i]):
                    entity_names[mids[i]] = set()
                entity_names[mids[i]].add(names[i].strip('"'))

    # print >> sys.stderr, entity_names
    return entity_names

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

# Open subprocess to evaluate queries
uschema_file = "src/lisp/universal_schema.lisp"
if model_type.endswith("_graphs"):
    if model_type != "ensemble_graphs":
        model_type = model_type.replace("_graphs", "")
    uschema_file = "src/lisp/universal_schema_with_graphs.lisp"

process = None
if model_type == "us":
    command = '''sbt "runMain com.jayantkrish.jklol.lisp.cli.AmbLisp --args '%s' src/lisp/environment.lisp %s %s %s src/lisp/run_universal_schema.lisp --interactive --noPrintOptions"''' % (model_filename, entity_file, word_file, uschema_file)
elif model_type == "ccg":
    command = '''sbt "runMain com.jayantkrish.jklol.lisp.cli.AmbLisp src/lisp/environment.lisp %s %s %s %s src/lisp/run_universal_schema_baseline.lisp --interactive --noPrintOptions"''' % (baseline_model_filename, entity_file, word_file, uschema_file)
elif model_type == "ensemble":
    command = '''sbt "runMain com.jayantkrish.jklol.lisp.cli.AmbLisp --args '%s' src/lisp/environment.lisp %s %s %s %s src/lisp/run_ensemble.lisp --interactive --noPrintOptions"''' % (model_filename, baseline_model_filename, entity_file, word_file, uschema_file)
elif model_type == "ensemble_graphs":
    command = '''sbt "runMain com.jayantkrish.jklol.lisp.cli.AmbLisp --args '%s' src/lisp/environment.lisp %s %s %s %s src/lisp/run_ensemble_with_graphs.lisp --interactive --noPrintOptions"''' % (model_filename, baseline_model_filename, entity_file, word_file, uschema_file)

if DEBUG:
    print >> sys.stderr, "Opening process..."
    print >> sys.stderr, "Command:", command

process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE)

fcntl.fcntl(process.stdout.fileno(), fcntl.F_SETFL, os.O_NONBLOCK)

output = read_until_prompt()
print output

entity_names = read_entity_names(all_data_file)

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
