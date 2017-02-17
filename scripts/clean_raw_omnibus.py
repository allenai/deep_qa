# -*- coding: utf-8 -*-
"""
This script takes as input raw TSV files from the Omnibus dataset and
preprocesses them to be compatible with the deep_qa pipeline.
"""
import logging
import os
import csv

from argparse import ArgumentParser
import pandas

logger = logging.getLogger(__name__) # pylint: disable=invalid-name


def main():
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format)
    parser = ArgumentParser(description=("Transform a raw Omnibus TSV "
                                         "to the format that the pipeline "
                                         "expects."))
    parser.add_argument('input_csv', nargs='+',
                        metavar="<input_csv>", type=str,
                        help=("Path of TSV files to clean up. Pass in "
                              "as many as you want, and the output "
                              "will be a concatenation of them "
                              "written to <last_input_csv>.clean"))

    arguments = parser.parse_args()
    all_clean_file_rows = []
    for omnibus_file in arguments.input_csv:
        all_clean_file_rows.extend(clean_omnibus_csv(omnibus_file))
    # turn the list of rows into a dataframe, and write to TSV
    dataframe = pandas.DataFrame(all_clean_file_rows)
    folder, filename = os.path.split(arguments.input_csv[-1])
    outdirectory = folder + "/cleaned/"
    os.makedirs(outdirectory, exist_ok=True)
    outpath = outdirectory + filename + ".clean"
    logger.info("Saving cleaned file to %s", outpath)
    dataframe.to_csv(outpath, encoding="utf-8", index=False,
                     sep="\t", header=False,
                     quoting=csv.QUOTE_NONE)


def clean_omnibus_csv(omnibus_file_path):
    logger.info("cleaning up %s", omnibus_file_path)
    # open the file as a csv
    dataframe = pandas.read_csv(omnibus_file_path, sep="\t",
                                encoding='utf-8', header=None,
                                quoting=csv.QUOTE_NONE)
    dataframe_trimmed = dataframe[[3, 9]]
    clean_rows = dataframe_trimmed.values.tolist()
    return clean_rows

if __name__ == '__main__':
    main()
