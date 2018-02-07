import argparse
import os.path
from pyparsing import Word, hexnums, WordEnd
from pyparsing import Optional, alphas, alphanums
from operator import add

from pyspark import SparkContext

def opcode_detect(asm_line):
    pattern = re.compile(r'([\s])([A-F0-9]{2})([\s]+)([a-z]+)([\s+])')
    pattern_list = pattern.findall(str(asm_line))
    if len(pattern_list)!=0:
        opcode = [item[3] for item in pattern_list][0]
    else: opcode = None
    return opcode

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "CSCI 8360 Project 2",
        epilog = "answer key", add_help = "How to use",
        prog = "python opcode_extracting.py [asm_files_path]")

    # Required args
    parser.add_argument("asm-path",help = "Directory of .asm files")

    args = vars(parser.parse_args())
    sc = SparkContext()
    path = args['asm-path']

    # Read in the files to RDD
    rdd_asm = sc.wholeTextFiles(path)

    # format keys and seperate contents
    rdd_asm = rdd_asm.map(lambda x: (x[0].replace('file:'+os.path.abspath(path)+'/', ''), x[1]))\
                    .map(lambda x: (x[0].replace('.asm', ''), x[1]))
    rdd_asm = rdd_asm.map(lambda x: (x[0], x[1].split('\n')))
    rdd_asm = rdd_asm.flatMapValues(lambda x: x) # RDD([(file_name, line_of_asm), ...])

    # Opcode 1-gram
    rdd_opcode = rdd_asm.map(lambda x: ((x[0], opcode_detect(x[1])), 1))
    rdd_opcode = rdd_opcode.filter(lambda x: x[0][1]!=None).reduceByKey(add)

    print(rdd_opcode.sortBy(lambda x: x[1]).take(200))
