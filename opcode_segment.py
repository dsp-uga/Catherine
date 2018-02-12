# python packages
import argparse
import os.path
import re
from operator import add
from math import log

# pyspark packages
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import NGram
# from pyspark.ml.linalg import Vectors
# from pyspark.ml.feature import StringIndexer
# from pyspark.ml.classification import RandomForestClassifier

def opcode_detect(asm_line):
    """
    Detects opcodes in each line in .asm file.
    Returns opcodes if opcodes exist or returns None.
    """
    pattern = re.compile(r'([\s])([A-F0-9]{2})([\s]+)([a-z]+)([\s+])')
    pattern_list = pattern.findall(str(asm_line))
    if len(pattern_list)!=0:
        opcode = [item[3] for item in pattern_list][0]
    else: opcode = None
    return opcode


def opcode_ngram(df_opcode, N):
    """
    Generates n-grams opcode by opcode data frame.
    Returns n-grams opcode in RDD((filename, n-gram), total_counts)
    """
    ngrams = NGram(n = N, inputCol = "opcode", outputCol = "ngrams")
    df_ngrams = ngrams.transform(df_opcode)
    rdd_ngrams = df_ngrams.select("filename", "ngrams").rdd.map(tuple)\
                        .flatMapValues(lambda x: x)\
                        .map(lambda x: ((x[0], x[1]), 1))\
                        .reduceByKey(add)
    return rdd_ngrams

def opcode_ngrams_combine(df_opcode, N):
    """
    Combines 2-grams, ..., N-grams opcode RDD
    >>> ((filename, opcode_ngrams), count)
    ordered by the count
    """
    rdd_ngrams_combine = opcode_ngram(df_opcode, 2)
    for n in range(2, N):
        rdd_ngrams = opcode_ngram(df_opcode, n+1)
        rdd_ngrams_combine = rdd_ngrams_combine.union(rdd_ngrams)
    rdd_ngrams_combine = rdd_ngrams_combine.sortBy(lambda x: x[1], ascending = False)
    return rdd_ngrams_combine

def segment_detect(asm_line):
    """
    Detects segment words in each line in .asm file.
    Returns segment words if exist or returns None.
    """
    pattern = re.compile(r'(.)([A-Za-z]+)(:)([0-9]{8})([\s+])')
    pattern_list = pattern.findall(str(asm_line))
    if len(pattern_list)!=0:
        segment = [item[1] for item in pattern_list][0]
    else: segment = None
    return segment

def feature_IDF(rdd_feature_cnt, top_n):
    """
    Calculate the N features with top N IDF with the format
    >>> (feature, IDF)
    """
    rdd_feature_distinct = rdd_feature_cnt.map(lambda x: x[0][1]).distinct().sortBy(lambda x: x)
    cnt_feature = rdd_feature_distinct.count()
    rdd_feature_IDF = rdd_feature_cnt.map(lambda x: (x[0][1], 1))\
                    .reduceByKey(add)\
                    .map(lambda x: (x[0], log(cnt_feature/x[1])))\
                    .sortBy(lambda x: x[1], ascending = False)\
                    .zipWithIndex().filter(lambda x: x[1] < top_n)\
                    .map(lambda x: x[0])
    return rdd_feature_IDF

def fillna(value):
    """
    Fills NA values after .join() or .leftOuterJoin().
    """
    if value != None:
        return value
    else:
        return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "CSCI 8360 Project 2",
        epilog = "answer key", add_help = "How to use",
        prog = "python opcode_segment.py [txt_files_path] [asm_files_path]")

    # Required args
    parser.add_argument("file-path", help = "Directory of .txt files")
    parser.add_argument("asm-path", help = "Directory of .asm files")
    # Optional args
    parser.add_argument("-s", "--size", choices = ["small", "large"], default = "small",
        help = "Size to the selected file: \"small\", \"large\" [Default: \"small\"]")

    args = vars(parser.parse_args())
    sc = SparkContext()
    spark = SparkSession.builder.master("local").appName("Word Count")\
                        .config("spark.some.config.option", "some-value").getOrCreate()
    file_path = args['file-path']
    asm_path = args['asm-path']
    if args['size'] == "small": size = '_small'
    if args['size'] == "large": size = ''

    # .txt files
    rdd_Xtrain = sc.textFile(file_path + 'X'+size+'_train.txt').zipWithIndex().map(lambda x: (x[1], x[0]))
    rdd_ytrain = sc.textFile(file_path + 'y'+size+'_train.txt').zipWithIndex().map(lambda x: (x[1], x[0]))
    rdd_train = rdd_Xtrain.join(rdd_ytrain).sortByKey().map(lambda x: x[1])
    rdd_Xtest = sc.textFile(file_path + 'X'+size+'_test.txt')

    # .asm files
    # Read in the files to RDD (sc.wholeTextFiles)
    rdd_asm = sc.wholeTextFiles(asm_path)\
                .map(lambda x: (x[0].replace('file:'+os.path.abspath(asm_path)+'/', ''), x[1]))\
                .map(lambda x: (x[0].replace('.asm', ''), x[1]))
    rdd_asm = rdd_Xtrain.map(lambda x: (x[0], x[1])).leftOuterJoin(rdd_asm).map(lambda x: (x[0], x[1][1]))\
                        .map(lambda x: (x[0], x[1].split('\n'))).flatMapValues(lambda x: x)

    # --testing small sets--
    # rdd_asm = sc.wholeTextFiles(asm_path)\
    #             .map(lambda x: (x[0].replace('file:'+os.path.abspath(asm_path)+'/', ''), x[1]))\
    #             .map(lambda x: (x[0].replace('.asm', ''), x[1]))\
    #             .map(lambda x: (x[0], x[1].split('\n'))).flatMapValues(lambda x: x)
    # rdd_Xtrain = rdd_asm.map(lambda x: x[0]).distinct()
    # l = rdd_Xtrain.collect()
    # rdd_train = rdd_train.filter(lambda x: x[0] in l)
    # --testing small sets--

    # Opcodes
    # >> (filename, [opcodes_list])
    rdd_opcode_list = rdd_asm.map(lambda x: (x[0], opcode_detect(x[1])))\
                        .filter(lambda x: x[1]!=None)\
                        .groupByKey().map(lambda x: (x[0], list(x[1])))
    df_opcode = spark.createDataFrame(rdd_opcode_list).toDF("filename", "opcode")
    # >> ((filename, opcode_ngrams), count)
    rdd_opcode_cnt = opcode_ngrams_combine(df_opcode, 4)
    # Top 1800 features
    rdd_opcode_IDF = feature_IDF(rdd_opcode_cnt, 1000)
    rdd_opcode_cnt_r = rdd_opcode_cnt.map(lambda x: (x[0][1], (x[0][0], x[1])))\
                            .leftOuterJoin(rdd_opcode_IDF).filter(lambda x: x[1][1]!=None)\
                            .map(lambda x: ((x[1][0][0], x[0]), x[1][0][1]))
    rdd_opcode = rdd_Xtrain.cartesian(rdd_opcode_cnt_r.map(lambda x: x[0][1]).distinct())\
                            .map(lambda x: (x,1)).leftOuterJoin(rdd_opcode_cnt_r)\
                            .map(lambda x: (x[0][0], (x[0][1], fillna(x[1][1]))))\
                            .sortBy(lambda x: (x[0], x[1][0]), ascending = False)
    # print('\n'.join(rdd_opcode.map(lambda x: str((x[0], x[1]))).take(50)))

    # Segment counts
    # >> ((filename, segment), count)
    rdd_segment_cnt = rdd_asm.map(lambda x: ((x[0], segment_detect(x[1])),1))\
                        .filter(lambda x: x[0][1]!=None)\
                        .filter(lambda x: 'EADER' not in x[0][1])\
                        .reduceByKey(add)
    # Top 200 features
    rdd_segment_IDF = feature_IDF(rdd_segment_cnt, 150)
    rdd_segment_cnt_r = rdd_segment_cnt.map(lambda x: (x[0][1], (x[0][0], x[1])))\
                            .leftOuterJoin(rdd_segment_IDF).filter(lambda x: x[1][1]!=None)\
                            .map(lambda x: ((x[1][0][0], x[0]), x[1][0][1]))
    rdd_segment = rdd_Xtrain.cartesian(rdd_segment_cnt_r.map(lambda x: x[0][1]).distinct())\
                            .map(lambda x: (x,1)).leftOuterJoin(rdd_segment_cnt_r)\
                            .map(lambda x: (x[0][0], (x[0][1], fillna(x[1][1]))))\
                            .sortBy(lambda x: (x[0], x[1][0]), ascending = False)
    # print('\n'.join(rdd_segment.map(lambda x: str((x[0], x[1]))).take(50)))

    # Combine two features
    # >> ((filename, label), (opcode_cnt1, opcode_cnt2, ..., segment_cnt1, ...))
    rdd_features_distinct = rdd_opcode.map(lambda x: x[1][0]).distinct().sortBy(lambda x: x, ascending = False)\
                                .union(rdd_segment.map(lambda x: x[1][0]).distinct().sortBy(lambda x: x, ascending = False))
    rdd_features = rdd_opcode.union(rdd_segment).leftOuterJoin(rdd_train)\
                    .map(lambda x: ((x[0], x[1][1]), x[1][0][1]))\
                    .groupByKey().map(lambda x: (x[0][0], x[0][1],)+ tuple(x[1]))
    # print('\n'.join(rdd_features.map(lambda x: str(x)).take(2)))
    # df_features = spark.createDataFrame(rdd_features)
    # df_features.show()
