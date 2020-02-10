package com.competitors

import breeze.linalg.{Axis, DenseMatrix, DenseVector, argmax}
import com.nmf.NMF
import com.nmf.mlmatrix.{RowPartition, RowPartitionedMatrix}
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd._
import com.utils.ExternalMetrics
import org.apache.spark.SparkContext
import org.apache.spark.mllib.clustering.{GaussianMixture}


abstract class Competitors(methodName: String, numClass: Int) {
    type inputRDD
    var accuracy = -1d
    var normalizedMI = -1d
    var fMeasure = -1d
    var elapsedTime = 0d

    def fit(sc: SparkContext, X: inputRDD): RDD[Int]

    def evaluation(sc: SparkContext, X: inputRDD, trueLabels: RDD[Int]): Double = {
        val startTime = System.currentTimeMillis()
        val predLabels = fit(sc, X)
        elapsedTime = (System.currentTimeMillis() - startTime) / 1000d
        val em = new ExternalMetrics(trueLabels, predLabels)
        normalizedMI = em.normalizedMI()
        accuracy = em.accuracy()
        fMeasure = em.fMeasure()
        elapsedTime
    }

    def printMetrics(): Unit = {
        println("methodName=%s" format methodName)
        if (normalizedMI > 0) println("NMI=%.4f" format normalizedMI)
        if (accuracy > 0) println("Accuracy=%.4f" format accuracy)
        if (fMeasure > 0) println("FMeasure=%.4f" format fMeasure)
        println("Total iteration Time = " + elapsedTime)
    }
}


class EvalKMeans(methodName: String = "KMeans", numClass: Int) extends Competitors(methodName: String, numClass: Int) {
    type inputRDD = RDD[Vector]
    override def fit(sc: SparkContext, X: inputRDD): RDD[Int] = {
        val clusters = KMeans.train(X, numClass, 100, initializationMode = "k-means||")
        val predLabels = clusters.predict(X)
        predLabels
    }
}

class EvalNMF(methodName: String = "NMF", numClass: Int) extends Competitors(methodName: String, numClass: Int) {
    type inputRDD = RowPartitionedMatrix
    override def fit(sc: SparkContext, X: inputRDD): RDD[Int] = {
        val nmfModel = new NMF()
        val nmfModel_res = nmfModel.fullNMF(X, numClass)

        val W = nmfModel_res._2
        val predLabel = argmax(W, Axis._1)
        sc.parallelize(predLabel.toArray)
    }
}

class EvalGMM(methodName: String = "GMM", numClass: Int) extends Competitors(methodName: String, numClass: Int) {
    type inputRDD = RDD[Vector]

    override def fit(sc: SparkContext, X: inputRDD): RDD[Int] = {
        //val df = ss.createDataset(X)
        val gmm = new GaussianMixture().setK(numClass).setMaxIterations(50).run(X)
        val predLabel = gmm.predict(X)
        predLabel
    }
}
