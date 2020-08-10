package com.competitors

import breeze.linalg.{Axis, DenseMatrix, DenseVector, argmax}
import com.nmf.mlmatrix.{RowPartition, RowPartitionedMatrix}
import com.utils.{AddNoiseMat, AddNoiseVector, NoiseSituation}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import scopt.OParser

case class CompareConfig(
                            input: String = null,
                            numClass: Int = 2,
                            hasLabel: Boolean = false,
                            delimiter: String = ",",
                            node: Int = 1,
                            seed: Long = 0,
                            std: Double = -1
                        )


object CompareMethods {

    def process(sc: SparkContext, config: CompareConfig, partitions: Int):
      (RDD[Vector], RDD[Int]) = {

        def text2arr(text: String, delimiter: String, hasLabel: Boolean):
        (Vector, Int) = {
            val arr = text.split(delimiter)
            if (hasLabel) {
                val label = arr(0).toDouble
                val arr_out = Vectors.dense(arr.slice(1, arr.length).map(_.toDouble))
                (arr_out, label.toInt)
            } else {
                val label = -1
                val arr_out = Vectors.dense(arr.map(_.toDouble))
                (arr_out, label)
            }
        }
        val X = sc.textFile(config.input).repartition(partitions)
        val res = X.map(x => text2arr(x, config.delimiter, config.hasLabel))
        val dataMat = res.map(_._1)
        val trueLabels = res.map(_._2)
//        val features = X.mapPartitionsWithIndex{ (index, iter) =>
//            val partition = iter.map(x => text2arr(x, config.delimiter, config.hasLabel)._1)
//            Array((index, partition)).iterator
//        }
//        val dataMat = features.map(x => (x._1, x._2.toList))
//        val labels = X.mapPartitionsWithIndex{ (index, iter) =>
//            val partition = iter.map(x => text2arr(x, config.delimiter, config.hasLabel)._2)
//            Array((index, partition)).iterator
//        }
//        val trueLabels = labels.map(x => (x._1, x._2.toList))
        dataMat.cache()
        (dataMat, trueLabels)
    }

    def main(args: Array[String]): Unit = {
        val builder = OParser.builder[CompareConfig]
        val parser1 = {
            import builder._
            OParser.sequence(
                programName("CompareMethods"),
                head("CompareMethods", "0.1"),
                opt[String]('f', name="input").required().action((x, c) => c.copy(input = x)).
                  text("path/to/input file"),
                opt[Int]('c', name="numClass").required().action((x, c) => c.copy(numClass = x)).
                  text("Number of class"),
                opt[String](name = "delimiter").optional().action((x, c) => c.copy(delimiter = x)).
                  text("Delimiter of input file."),
                opt[Int]('n', name = "node").required().action((x, c) => c.copy(node = x)).
                    text("Number of nodes to be used."),
                opt[Unit](name = "hasLabel").optional().action((_, c) => c.copy(hasLabel = true)).
                  text("Contains label"),
                opt[Double](name = "std").optional.action((x, c) => c.copy(std = x)).
                    text("proportion of noise")
            )
        }
        OParser.parse(parser1, args, CompareConfig()) match {
            case Some(config) => {
                println("Input file: " + config.input)
                val spark = SparkSession.builder().getOrCreate()
                val sc = spark.sparkContext
                sc.setLogLevel("WARN")
                val dataset = process(sc, config, config.node)
                var X = dataset._1
                if (config.std > 0) {
                    val s  = NoiseSituation.getComSituation(config.std)
                    X = X.map(x => AddNoiseVector.addGaussianMixture(x, s.stdVector, s.mixtureProp)).cache()
                    println(">>Add noise to data. sd=%.2f" format config.std)

                }
                val trueLabels = dataset._2
                print("====================KMeans==================\n")
                val kMeansMoodel = new EvalKMeans(numClass = config.numClass)
                kMeansMoodel.evaluation(sc, X, trueLabels)
                kMeansMoodel.printMetrics()
                print("====================NMF======================\n")
                val nmfModel = new EvalNMF(numClass = config.numClass)
                val numcols = X.first().size
                def buildRowBlock(iter: Iterator[Vector]): Iterator[RowPartition] = {
                    val vectorList = iter.toList
                    val mat = DenseMatrix.zeros[Double](vectorList.length, numcols)
                    vectorList.zipWithIndex.foreach(x => mat(x._2, ::) := DenseVector(x._1.toArray).t)
                    Array(RowPartition(mat)).toIterator
                }
                val rpmX = new RowPartitionedMatrix(X.mapPartitions(buildRowBlock, true))
                nmfModel.evaluation(sc, rpmX, trueLabels)
                nmfModel.printMetrics()
            }
            case None =>
                println("Input arguments are not valid")
                System.exit(0)
        }
    }
}