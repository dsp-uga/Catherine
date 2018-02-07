import argparse
import os.path
from pyparsing import Word, hexnums, WordEnd
from pyparsing import Optional, alphas, alphanums
from operator import add

from pyspark import SparkContext

# only drag those starts with .text:
def opcode_detect(asm_line):
    # use WordEnd() to avoid parsing leading a-f of non-hex numbers as a hex
    if asm_line.startswith('.text:'):
        hex_integer = Word(hexnums) + WordEnd()
        line = ".text:" + hex_integer + Optional((hex_integer*(1,))("instructions") + Word(alphas,alphanums)("opcode"))
        result = line.parseString(asm_line)
        if "opcode" in result:
            return result.opcode
    else:
        return None


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

    print(rdd_opcode.take(200))
