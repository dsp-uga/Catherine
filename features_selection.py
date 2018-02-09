# python packages
import argparse
import os.path
import re
from operator import add

# pyspark packages
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import NGram
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import RandomForestClassifier

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

def segment_detect(asm_line):
    """
    Detects segment words in each line in .asm file.
    Returns segment words if exist or returns None.
    """
    pattern = re.compile(r'(.)([A-Za-z]+)(:)([0-9]{8})')
    pattern_list = pattern.findall(str(asm_line))
    if len(pattern_list)!=0:
        segment = [item[1] for item in pattern_list][0]
    else: segment = None
    return segment

def fillna(value):
    """
    Fills NA values after .join() or .leftOuterJoin().
    """
    if value != None:
        return value
    else:
        return 0

def RF_features_select(df_features, n=3, s=50):
    stringIndexer = StringIndexer(inputCol = "label", outputCol = "indexed")
    si_model = stringIndexer.fit(df_features)
    td = si_model.transform(df_features)
    rf = RandomForestClassifier(numTrees = n, maxDepth = 2, labelCol = "indexed", seed = s)
    model = rf.fit(td)
    feature_imp = model.featureImportances
    return feature_imp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "CSCI 8360 Project 2",
        epilog = "answer key", add_help = "How to use",
        prog = "python opcode_extracting.py [txt_files_path] [asm_files_path]")

    # Required args
    parser.add_argument("file-path", help = "Directory of .txt files")
    parser.add_argument("asm-path", help = "Directory of .asm files")

    args = vars(parser.parse_args())
    sc = SparkContext()
    spark = SparkSession.builder.master("local").appName("Word Count")\
                        .config("spark.some.config.option", "some-value").getOrCreate()
    file_path = args['file-path']
    asm_path = args['asm-path']

    # .txt files
    rdd_Xtrain = sc.textFile(file_path + 'X_small_train.txt').zipWithIndex().map(lambda x: (x[1], x[0]))
    rdd_ytrain = sc.textFile(file_path + 'y_small_train.txt').zipWithIndex().map(lambda x: (x[1], x[0]))
    rdd_train = rdd_Xtrain.join(rdd_ytrain).sortByKey().map(lambda x: x[1])

    # .asm files
    # Read in the files to RDD
    rdd_asm = sc.wholeTextFiles(asm_path)
    # Read in the files to RDD (sc.textFiles)
    # -- testing --

    # format keys and seperate contents
    rdd_asm = rdd_asm.map(lambda x: (x[0].replace('file:'+os.path.abspath(asm_path)+'/', ''), x[1]))\
                    .map(lambda x: (x[0].replace('.asm', ''), x[1]))\
                    .map(lambda x: (x[0], x[1].split('\n')))\
                    .flatMapValues(lambda x: x) # RDD([(file_name, line_of_asm), ...])

    # --testing small sets--
    rdd_mini_Xtrain = rdd_asm.map(lambda x: x[0]).distinct()
    l = rdd_mini_Xtrain.collect()
    rdd_mini_train = rdd_train.filter(lambda x: x[0] in l)
    # --testing small sets--

    # Opcodes
    rdd_opcode = rdd_asm.map(lambda x: (x[0], opcode_detect(x[1])))\
                        .filter(lambda x: x[1]!=None)
    rdd_opcode_list = rdd_opcode.groupByKey().map(lambda x: (x[0], list(x[1])))
    df_opcode = spark.createDataFrame(rdd_opcode_list).toDF("filename", "opcode")
    rdd_2grams = opcode_ngram(df_opcode, 2)
    rdd_3grams = opcode_ngram(df_opcode, 3)
    rdd_4grams = opcode_ngram(df_opcode, 4)

    # Segment counts
    # ((filename, segment), count)
    rdd_segment_cnt = rdd_asm.map(lambda x: ((x[0], segment_detect(x[1])),1))\
                        .filter(lambda x: x[0][1]!=None)\
                        .filter(lambda x: 'EADER' not in x[0][1])\
                        .reduceByKey(add)
    # (segment), ordered distinct segments
    rdd_segment_distinct = rdd_segment_cnt.map(lambda x: x[0][1]).distinct().sortBy(lambda x: x)
    rdd_segment_distinct_ind = rdd_segment_distinct.zipWithIndex().map(lambda x: (x[1], x[0]))
    # (filename, (segment, count))
    rdd_file_segment_cnt = rdd_mini_Xtrain.cartesian(rdd_segment_distinct)\
                                    .map(lambda x: (x, 1))\
                                    .leftOuterJoin(rdd_segment_cnt)\
                                    .map(lambda x: (x[0][0], (x[0][1], fillna(x[1][1]))))
    # ((filename, label), segment_cnt_list)
    rdd_file_label_seglist = rdd_mini_train.leftOuterJoin(rdd_file_segment_cnt)\
                                    .map(lambda x: ((x[0], x[1][0]), x[1][1]))\
                                    .sortBy(lambda x: x[1][0], ascending = True)\
                                    .map(lambda x: (x[0], x[1][1]))\
                                    .groupByKey().map(lambda x: (x[0], list(x[1])))
    # RF Features Selection
    data_segment = rdd_file_label_seglist.map(lambda x: (x[0][1], Vectors.dense(x[1])))
    df_segment = spark.createDataFrame(data_segment).toDF("label", "features")
    # (index, feature_importance)
    rdd_segment_imp = sc.parallelize(RF_features_select(df_segment)).zipWithIndex()\
                        .map(lambda x: (x[1], x[0]))
    # (choosed segment)
    rdd_segment_choose = rdd_segment_distinct_ind.leftOuterJoin(rdd_segment_imp)\
                            .filter(lambda x: x[1][1]!=0).map(lambda x: x[1][0])

    print(rdd_segment_choose.collect())
