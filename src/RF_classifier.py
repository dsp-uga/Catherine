# python packages
import argparse
import os.path
import re
import numpy as np
from operator import add

# pyspark packages
from pyspark import SparkContext
from pyspark.sql import SparkSession

from pyspark.ml.linalg import Vectors
from pyspark.sql.types import DoubleType
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer

def _format_(rdd, feature_path):
    f = feature_path.lower()
    if 'train' in f:
        if 'segment' in f: rdd_fea = rdd.map(lambda x: ((x[0], x[1], int(x[2])), x[3]))
        if 'file' in f: rdd_fea = rdd.map(lambda x: ((x[0], x[1], x[5]), Vectors.dense(list((x[2], x[3], x[4])))))
        if 'gram' in f: rdd_fea = rdd.map(lambda x: ((x[0], x[1], x[3]), Vectors.dense(list(x[2].toArray()))))
    if 'test' in f:
        if 'segment' in f: rdd_fea = rdd.map(lambda x: ((x[1], x[0]), x[2]))
        if 'file' in f: rdd_fea = rdd.map(lambda x: ((x[0], x[1]), Vectors.dense(list((x[2], x[3], x[4])))))
        if 'gram' in f: rdd_fea = rdd.map(lambda x: ((x[0], x[1]), Vectors.dense(list(x[2].toArray()))))
    return rdd_fea

def RF_format(df):
    # Input >> DF(docid, hash, label, vector.dense(features))
    stringIndexer = StringIndexer(inputCol = "hash", outputCol = "indexed")
    si_model = stringIndexer.fit(df)
    td = si_model.transform(df)
    return td

def RF_model(td, n, m, s = 50):
    # td_new = change_column_datatype(td, "label", DoubleType)
    td_new = td.withColumn("label", td["label"].cast(DoubleType()))
    rf = RandomForestClassifier(numTrees = n, maxDepth = m, maxBins=32, labelCol = "label", seed = s)
    model = rf.fit(td_new)
    return model

def cal_accuracy(label_list, pred_list):
    cnt = 0
    ttl = len(label_list)
    for doc in range(ttl):
        pred_list[doc] = str(pred_list[doc])
        if pred_list[doc] == label_list[doc]: cnt += 1
    accuracy = cnt / ttl
    return accuracy

def output_file(output_pred, output_path):
    outF = open(output_path, "w")
    textList = '\n'.join(output_pred)
    outF.writelines(textList)
    outF.close()
    return 'output_file has been saved!'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "CSCI 8360 Project 2",
        epilog = "answer key", add_help = "How to use",
        prog = "python RF_classifier.py [parquet_paths]")

    # Required args
    parser.add_argument("parquets", help = "Paths to parquets, please separate them by comma.")
    # Optional args
    parser.add_argument("-n", "--treenum", default = 10, type = int,
        help = "Number of trees in random forest classifier.")
    parser.add_argument("-m", "--maxdepth", default = 5, type = int,
        help = "Maximum depth of braches in random forest classifier.")

    args = vars(parser.parse_args())
    sc = SparkContext()
    spark = SparkSession.builder.master("local").appName("Word Count")\
                        .config("spark.some.config.option", "some-value").getOrCreate()
    paquets = args['parquets']
    N = args['treenum']
    M = args['maxdepth']

    paths = paquets.split(',')
    f = len(paths)/2
    if f >= 1: fea1_train_path, fea1_test_path = paths[0:2]
    if f >= 2: fea2_train_path, fea2_test_path = paths[2:4]
    if f >= 3: fea3_train_path, fea3_test_path = paths[4:6]
    if f >= 4: fea4_train_path, fea4_test_path = paths[6:8]
    output_path = 'outputs/'

    print(fea1_train_path)
    if '1-gram' in fea1_train_path.lower(): print('YEP!')
    else: print('WTF')


    ## one feature
    if f == 1:
    # ------------------------------------------------------------------------
        rdd_train = spark.read.parquet(fea1_train_path).rdd.map(tuple)
        rdd_test = spark.read.parquet(fea1_test_path).rdd.map(tuple)

        if 'segment' in fea1_train_path.lower():
            rdd_train = rdd_train
            rdd_test = rdd_test.map(lambda x: (x[1], x[0], x[2]))
        if 'file' in fea1_train_path.lower():
            rdd_train = rdd_train.map(lambda x: (x[0], x[1], x[5], Vectors.dense(list(x[2], x[3], x[4]))))
            rdd_test = rdd_test.map(lambda x: (x[0], x[1], Vectors.dense(list(x[2], x[3], x[4]))))
        # if '1-gram' or '2-gram' in fea1_train_path.lower():
        #     rdd_train = rdd_train.map(lambda x: (x[0], x[1], x[3], Vectors.dense(list(x[2].toArray()))))
        #     rdd_test = rdd_test.map(lambda x: (x[0], x[1], Vectors.dense(list(x[2].toArray()))))

        print(rdd_train.take(1))

        print('***** Training Dataframe ******************************************')
        df_train = spark.createDataFrame(rdd_train, ['docid', 'hash', 'label', 'features'])
        df_train.show()
        print('***** Testing Dataframe ******************************************')
        df_test = spark.createDataFrame(rdd_test, ['docid', 'hash', 'features'])
        df_test.show()
    # ------------------------------------------------------------------------

    ## two more features
    if f == 2:
    # ------------------------------------------------------------------------
        fea1_train_rdd = spark.read.parquet(fea1_train_path).rdd.map(tuple)
        fea1_train = _format_(fea1_train_rdd, fea1_train_path)
        fea2_train_rdd = spark.read.parquet(fea2_train_path).rdd.map(tuple)
        fea2_train = _format_(fea2_train_rdd, fea2_train_path)
        rdd_train = fea1_train.leftOuterJoin(fea2_train)\
                        .map(lambda x: (x[0][0], x[0][1], x[0][2], Vectors.dense(list(x[1][0]) + list(x[1][1]))))

        fea1_test_rdd = spark.read.parquet(fea1_test_path).rdd.map(tuple)
        fea1_test = _format_(fea1_test_rdd, fea1_test_path)
        fea2_test_rdd = spark.read.parquet(fea2_test_path).rdd.map(tuple)
        fea2_test = _format_(fea2_test_rdd, fea2_test_path)
        rdd_test = fea1_test.leftOuterJoin(fea2_test)\
                        .map(lambda x: (x[0][0], x[0][1], Vectors.dense(list(x[1][0]) + list(x[1][1]))))

        print('***** Training Dataframe ******************************************')
        df_train = spark.createDataFrame(rdd_train, ['docid', 'hash', 'label', 'features'])
        df_train.show()
        print('***** Testing Dataframe ******************************************')
        df_test = spark.createDataFrame(rdd_test, ['docid', 'hash', 'features'])
        df_test.show()
    # ------------------------------------------------------------------------

    ## three more features
    if f == 3:
    # ------------------------------------------------------------------------
        fea1_train_rdd = spark.read.parquet(fea1_train_path).rdd.map(tuple)
        fea1_train = _format_(fea1_train_rdd, fea1_train_path)
        fea2_train_rdd = spark.read.parquet(fea2_train_path).rdd.map(tuple)
        fea2_train = _format_(fea2_train_rdd, fea2_train_path)
        fea3_train_rdd = spark.read.parquet(fea3_train_path).rdd.map(tuple)
        fea3_train = _format_(fea3_train_rdd, fea3_train_path)
        rdd_train = fea1_train.leftOuterJoin(fea2_train)\
                        .map(lambda x: (x[0], Vectors.dense(list(x[1][0]) + list(x[1][1]))))\
                        .leftOuterJoin(fea3_train)\
                        .map(lambda x: (x[0][0], x[0][1], x[0][2], Vectors.dense(list(x[1][0]) + list(x[1][1]))))

        fea1_test_rdd = spark.read.parquet(fea1_test_path).rdd.map(tuple)
        fea1_test = _format_(fea1_test_rdd, fea1_test_path)
        fea2_test_rdd = spark.read.parquet(fea2_test_path).rdd.map(tuple)
        fea2_test = _format_(fea2_test_rdd, fea2_test_path)
        fea3_test_rdd = spark.read.parquet(fea3_test_path).rdd.map(tuple)
        fea3_test = _format_(fea3_test_rdd, fea3_test_path)
        rdd_test = fea1_test.leftOuterJoin(fea2_test)\
                        .map(lambda x: (x[0], Vectors.dense(list(x[1][0]) + list(x[1][1]))))\
                        .leftOuterJoin(fea3_test)\
                        .map(lambda x: (x[0][0], x[0][1], Vectors.dense(list(x[1][0]) + list(x[1][1]))))

        print('***** Training Dataframe ******************************************')
        df_train = spark.createDataFrame(rdd_train, ['docid', 'hash', 'label', 'features'])
        df_train.show()
        print('***** Testing Dataframe ******************************************')
        df_test = spark.createDataFrame(rdd_test, ['docid', 'hash', 'features'])
        df_test.show()
    # ------------------------------------------------------------------------

    ## three more features
    if f == 4:
    # ------------------------------------------------------------------------
        fea1_train_rdd = spark.read.parquet(fea1_train_path).rdd.map(tuple)
        fea1_train = _format_(fea1_train_rdd, fea1_train_path)
        fea2_train_rdd = spark.read.parquet(fea2_train_path).rdd.map(tuple)
        fea2_train = _format_(fea2_train_rdd, fea2_train_path)
        fea3_train_rdd = spark.read.parquet(fea3_train_path).rdd.map(tuple)
        fea3_train = _format_(fea3_train_rdd, fea3_train_path)
        fea4_train_rdd = spark.read.parquet(fea4_train_path).rdd.map(tuple)
        fea4_train = _format_(fea4_train_rdd, fea4_train_path)
        rdd_train = fea1_train.leftOuterJoin(fea2_train).map(lambda x: (x[0], Vectors.dense(list(x[1][0]) + list(x[1][1]))))\
                        .leftOuterJoin(fea3_train).map(lambda x: (x[0], Vectors.dense(list(x[1][0]) + list(x[1][1]))))\
                        .leftOuterJoin(fea4_train).map(lambda x: (x[0][0], x[0][1], x[0][2], Vectors.dense(list(x[1][0]) + list(x[1][1]))))

        fea1_test_rdd = spark.read.parquet(fea1_test_path).rdd.map(tuple)
        fea1_test = _format_(fea1_test_rdd, fea1_test_path)
        fea2_test_rdd = spark.read.parquet(fea2_test_path).rdd.map(tuple)
        fea2_test = _format_(fea2_test_rdd, fea2_test_path)
        fea3_test_rdd = spark.read.parquet(fea3_test_path).rdd.map(tuple)
        fea3_test = _format_(fea3_test_rdd, fea3_test_path)
        fea4_test_rdd = spark.read.parquet(fea4_test_path).rdd.map(tuple)
        fea4_test = _format_(fea4_test_rdd, fea4_test_path)
        rdd_test = fea1_test.leftOuterJoin(fea2_test)\
                        .map(lambda x: (x[0], Vectors.dense(list(x[1][0]) + list(x[1][1]))))\
                        .leftOuterJoin(fea3_test)\
                        .map(lambda x: (x[0][0], x[0][1], Vectors.dense(list(x[1][0]) + list(x[1][1]))))
        rdd_test = fea1_test.leftOuterJoin(fea2_test).map(lambda x: (x[0], Vectors.dense(list(x[1][0]) + list(x[1][1]))))\
                        .leftOuterJoin(fea3_test).map(lambda x: (x[0], Vectors.dense(list(x[1][0]) + list(x[1][1]))))\
                        .leftOuterJoin(fea4_test).map(lambda x: (x[0][0], x[0][1], Vectors.dense(list(x[1][0]) + list(x[1][1]))))

        print('***** Training Dataframe ******************************************')
        df_train = spark.createDataFrame(rdd_train, ['docid', 'hash', 'label', 'features'])
        df_train.show()
        print('***** Testing Dataframe ******************************************')
        df_test = spark.createDataFrame(rdd_test, ['docid', 'hash', 'features'])
        df_test.show()
    # ------------------------------------------------------------------------

    # Classification
    ##########################################################################
    # Model & Prediction
    model = RF_model(RF_format(df_train), n=N, m=M)
    # pred = model.transform(RF_format(df_test))
    pred = model.transform(df_test)
    pred = pred.withColumn("prediction", pred["prediction"].cast("int"))
    pred.show()
    y_test = pred.select("docid", "prediction").rdd.map(tuple)\
                .sortByKey().map(lambda x: x[1]).collect()
    print(y_test[0:30])
    for i in range(len(y_test)):
        y_test[i] = str(y_test[i])

    # Accuracy
    # rdd_ytest = sc.textFile('files/y_small_test.txt')
    # accuracy = cal_accuracy(rdd_ytest.collect(), y_test)
    # print('Testing Accuracy: %.2f %%' % (accuracy*100))
    # print('**********************************************')

    # Output file
    output = output_path + 'prediction.txt'
    output_file(y_test, output)
