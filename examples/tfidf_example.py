from __future__ import print_function

__author__ = 'prem'
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.sql import SQLContext
from pyspark import SparkContext


if __name__ == "__main__":
    sc = SparkContext(appName="TfIdfExample")
    sqlContext = SQLContext(sc)

    # $example on$
    sentenceData = sqlContext.createDataFrame([
        (0, "Hi I heard about Spark"),
        (0, "I wish Java could use case classes"),
        (1, "Logistic regression models are neat")
    ], ["label", "sentence"])
    tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
    wordsData = tokenizer.transform(sentenceData)
    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
    featurizedData = hashingTF.transform(wordsData)
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idfModel = idf.fit(featurizedData)
    rescaledData = idfModel.transform(featurizedData)
    for features_label in rescaledData.select("features", "label").take(3):
        print(features_label)
    # $example off$

    sc.stop()
