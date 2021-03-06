# Malware Classification

This repository contains a Random Forest classifier implemented on malware classification which is completed on CSCI 8360, Data Science Practicum at the University of Georgia, Spring 2018.

This project uses the hexadecimal binaries as documents, and classify them into one of several possible malware families. The data are from the [Microsoft Malware Classification Challenge](https://www.kaggle.com/c/malware-classification/),  which consists of nearly half a terabyte of uncompressed data. The 9 classes of malware are as follows:

1. Ramnit
2. Lollipop
3. Kelihos_ver3
4. Vundo
5. Simda
6. Tracur
7. Kelihos_ver1
8. Obfuscator.ACY
9. Gatak

All the documents have a corresponding **hash**, **bytes file**, and **asm file**. All the hashes are listed in `X_train.txt` and `X_test.txt` documents, and the labels for documents in training set are in `y_train.txt` file. The previews of bytes files and asm files showed as follows:

### bytes file

<p align="center">
<img src="img/bytes_preview.png" />
</p>

- `00401000`: hexadecimal tokan, a line pointer and can be safely ignored
- `A4`, `AC`, `4A`: hexadecimal pairs, the code of the malware instance itself

### asm file

<p align="center">
<img src="img/asm_preview.png" />
</p>

- `text`: segments, included segments for containing instruction codes, segments for storing the data elements, and segments for keeping the program stack. (the first text of each line in asm file)
- `push`, `lea`, `xor`: opcodes (operation code), a machine language instruction that specifies the operation to be performed. (the first text after bytes of each line in asm file)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

  - [Python 3.6](https://www.python.org/downloads/release/python-360/)
  - [Anaconda](https://www.anaconda.com/)
  - [Apache Spark 2.2.1](http://spark.apache.org/)
  - [Pyspark 2.2.1](https://pypi.python.org/pypi/pyspark/2.2.1) - Python API for Apache Spark (`pyspark.ml`, `pyspark.sql`)
  - [Google Cloud Platform](https://cloud.google.com)

## Running the tests

You can run all the `.py` scripts via **python** or **spark-submit** on your local machine. Make sure to speciify the exact path of your spark-submit.

Notes that the dataset using in this case is extremely huge, reading the whole dataset, computing features, and implementing classification every time might take the whole day for only one result.

In this project, we separated the whole process into **feature extraction** and **classification** two parts. You are able to select those features that interest you and input them into the random forest classifier by instruction.

### Features Extraction

There are seven features extracted in this case. Read following description of how they were extracted and what script they are in:

#### Features

1. **bytes file size**
   - The file size is calculated using file_processor.py. File sizes for all files have been calculated using both urllib
     as well as sys library in spark for rdd. RDD approach takes a lot of memory
2. **asm file size**
   - The file size is calculated using file_processor.py.
3. **bytes and asm file size ratio**
   - The ratio of asm and bytes file is taken.
4. **unigram bytes (from bytes files)**
    - Idea behind calculating unigrams is that hexadecimal digits when converted to integer can take values between 
      0-256. So accordingly we create a vector of 256 size and make the count of each number which occurs
5. **bigram bytes (from bytes files)**
     Idea behind calculating unigrams is that hexadecimal digits when converted to integer can take values between 
      0-256*256 . So accordinly we create a vector of 256 cross 256 size and make the count each number which occurs
      However we reduce the feature size using the top 2000 elements for each row
6. **segment (from asm files)** - `segment_cnt.py`

    - Detected the segments in each asm file
    - Recorded the counts in each document
    - Resulted in 257 different segments.

7. **2-4 grams opcode (from asm files)** - `opcode_ngrams.py`

    - Detected the opcodes in each asm file
    - Selected those opcodes appeared only in 1/3 documents
    - Generated 2, 3, 4 grams opcodes by selected opcodes
    - Selected the important features by random forest classifier
    - Recorded the counts of each opcodes in each document

#### Running

```
$ python <feature_script>.py [file-directory] [bytes/asm_file-directory] [output-directory] [optional args]
```
```
$ usr/bin/spark-submit <feature_script>.py [file-directory] [bytes/asm_file-directory] [output-directory] [optional args]
```

#### Required Arguments

- `file-path`: Directory contains the input hash and label files

- `bytes-path` or `asm-path`: Directory contains the input `.bytes` or `.asm` files

- `output-path`: Directory to output files

#### Optional Arguments

- `-s`: Sizes to the selected file. (Default: `small`)

  `small`: selecting the small dataset containing 379 training files and 169 testing files.
  `large`: selecting the large dataset containing 8147 training files and 2721 testing files.

### Random Forest Classifier

We used the built-in random forest classifier in `pyspark.ml`. Input the `.parquet` files obtained in previous part, and the classifier will output a list of prediction of malware classes for each document.

#### Running

```
$ python RF_classifier.py [directories] [optional args]
```
```
$ usr/bin/spark-submit RF_classifier.py [directories] [optional args]
```

#### Required Arguments

 - `directories`: those directories containing the parquets files you generated from previous feature selection part. List every train and test directories in order and separate them by comma.
 e.g. `segment_train,segment_test,1-gram-train,1-gram_test`

#### Optional Arguments

 - `-n`: Number of trees in random forest classifier, default = 10
 - `-m`: Maximum depth of each branch in random forest classifier, default = 5

## Test Results

We resulted in accuracy of 98.97% by selecting segment and unigram bytes as our feature with 50 trees and 25 maximum depth for each branch in the classifier. The number of trees and maximum depth did influence the accuracy. See the following table for more combination of attempt:

(According to the process time of opcode in large dataset, we did not include opcodes in the discussion here, and we are still skeptical to the better result of adding opcodes in the classifier due to its sparse feature vectors.)

|Bytes Size|Asm Size|Size Ratio|Unigram|Bigram|Segment|Trees|Depth|Accuracy|
|----------|--------|----------|-------|------|-------|-----|-----|--------|
|          | v      |          |       |      |       | 10  | 5   |66.00%  |
|          |        |          |       |      | v     | 10  | 5   |87.10%  |
|          | v      |          |       |      | v     | 10  | 5   |90.00%  |
| v        | v      | v        |       |      | v     | 10  | 5   |93.16%  |
|          |        |          |       |      | v     | 50  | 25  |94.85%  |
|          | v      |          | v     | v    | v     | 10  | 5   |96.03%  |
|          | v      |          | v     | v    |       | 10  | 5   |96.14%  |
| v        | v      | v        | v     |      | v     | 10  | 8   |96.32%  |
|          | v      |          | v     |      | v     | 10  | 5   |96.58%  |
|          |        |          | v     |      | v     | 10  | 5   |96.83%  |
| v        | v      | v        | v     |      | v     | 25  | 10  |97.75%  |
|          |        |          | v     |      | v     | 25  | 10  |97.94%  |
| v        | v      | v        | v     |      | v     | 50  | 25  |98.64%  |
|          |        |          | v     |      | v     | 60  | 30  |98.75%  |
|          |        |          | v     |      | v     | 70  | 30  |98.75%  |
|          |        |          | v     |      | v     | 40  | 15  |98.78%  |
|          |        |          | v     |      | v     | 55  | 25  |98.82%  |
|          |        |          | v     |      | v     | 45  | 25  |98.93%  |
|          |        |          | v     |      | v     | 45  | 30  |98.93%  |
|          |        |          | v     |      | v     | 50  | 25  |98.97%  |
|          |        |          | v     |      | v     | 50  | 28  |98.97%  |

## Future Research

Random forest classifier can be a nice classifier to deal with sparse features (which is the reason we implement it while choosing opcode features), however, gradient boosting classifier might be a better way to work on dense features. To improve this classifier, we expect to further the project by adding in opcodes (after selecting by RF classifier), and implement gradient boosting classifier for all features we selected. Moreover, we attempt to add in images feature (image of bytes and asm files).

## Authors
(Ordered alphabetically)

- **I-Huei Ho** - [melanieihuei](https://github.com/melanieihuei)
- **Parya Jandaghi** - [parya-j](https://github.com/parya-j)
- **Vibodh Fenani** - [vibodh01](https://github.com/vibodh01)

See the [CONTRIBUTORS](CONTRIBUTORS.md) file for details.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
