#!/usr/bin/python

import sys

annotation_filenames = sys.argv[1:]

NUM_TO_ANNOTATE=30

mid_names = {}

def read_annotations_from_file(filename):
    annotations = {}
    query_strs = []
    query_text_strs = {}
    with open(filename, 'r') as annotation_file:
        query_str = None
        in_query = False
        true_mids = set([])
        false_mids = set([])
        unknown_mids = set([])
        num_mids = 0

        prev_line = None
        query_text = None
        for line in annotation_file:
            if line.startswith("(lambda"):
                query_str = line.strip()
                query_text = prev_line.strip()
            elif line.startswith("AP:"):
                annotations[query_str] = (true_mids, unknown_mids, false_mids)
                query_strs.append(query_str)
                query_text_strs[query_str] = query_text

                query_str = None
                true_mids = set([])
                false_mids = set([])
                unknown_mids = set([])
                num_mids = 0

            elif line.startswith("1 ") or line.startswith("0 ") or line.startswith("? "):
                parts = line.split(" ")
                annotation = parts[0]
                score = float(parts[1])
                mid = parts[2]
                names = " ".join(parts[3:]).strip()
                mid_names[mid] = names

                if num_mids < NUM_TO_ANNOTATE and score > 0:
                    if parts[0] == "1":
                        true_mids.add(mid)
                    elif parts[0] == "?":
                        unknown_mids.add(mid)
                    else:
                        false_mids.add(mid)

                num_mids += 1

            prev_line = line
    return annotations, query_strs, query_text_strs

annotations = []
query_strs = None
query_text = None
for annotation_filename in annotation_filenames:
    (file_annotations, query_strs, query_text) = read_annotations_from_file(annotation_filename)
    annotations.append(file_annotations)

for query_str in query_strs:
    true_mids = set([])
    unknown_mids = set([])
    false_mids = set([])

    for annotation in annotations:
        true_mids.update(annotation[query_str][0])
        unknown_mids.update(annotation[query_str][1])
        false_mids.update(annotation[query_str][2])

    print
    print query_text[query_str]
    print query_str

    url_fmt = "http://www.freebase.com%s"
    width = 10
    for true_mid in true_mids:
        print "1", true_mid.ljust(width), url_fmt % true_mid.ljust(width), mid_names[true_mid]

    for unknown_mid in unknown_mids:
        print "?", unknown_mid.ljust(width), url_fmt % unknown_mid.ljust(width), mid_names[unknown_mid]

    for false_mid in false_mids:
        print "0", false_mid.ljust(width), url_fmt % false_mid.ljust(width), mid_names[false_mid]
