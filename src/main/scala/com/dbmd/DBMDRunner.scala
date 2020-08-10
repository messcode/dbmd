package com.dbmd

import breeze.linalg.{DenseMatrix, DenseVector, min}
import com.utils.{AddNoiseMat, ExternalMetrics, MatUtil, NoiseSituation, TextConverter}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import scopt.OParser

case class DBMDConfig(
                         input: String = null,
                         node: Int = 1,
                         output: String = null,
                         rank: Int = 1,
                         lam: Double = 0.5,
                         hp: Double = 5.0,
                         maxIter: Int = 20,
                         hasLabel: Boolean = false,
                         delimiter: String = ",",
                         tol: Double = 1e-3,
                         cv: Boolean = false,
                         alpha: Double = 0.1d,
                         init_method: String = "randomCols",
                         std: Double = -1,
                         algorithm: String = "ADMM",
                         dropConstraints: Boolean = false,
                         repeat: Int = 1
                     )

object DBMDRunner {
    def process(sc: SparkContext, config: DBMDConfig): (RDD[((Int, Int), DenseMatrix[Double])], RDD[Int], DenseMatrix[Double]) = {
        // 1. load data
        def text2arr(text: String, delimiter: String, hasLabel: Boolean):
        (DenseVector[Double], Int) = {
            val arr = text.split(delimiter)
            if (hasLabel) {
                val label = arr(0).toDouble
                val arr_out = DenseVector(arr.slice(1, arr.length).map(_.toDouble))
                (arr_out, label.toInt)
            } else {
                val label = -1
                val arr_out = DenseVector(arr.map(_.toDouble))
                (arr_out, label)
            }
        }
        val X = sc.textFile(config.input).repartition(config.node) // repatition RDD to C slices
        X.cache()
        var m = X.first().split(config.delimiter).length
        if (config.hasLabel){
            m = m - 1
        }
        val W = DenseMatrix.zeros[Double](m, config.rank)
        val result = X.mapPartitionsWithIndex((index, iterX) => {
            val listX = iterX.toList
            val nc = listX.length
            val labels = new Array[Int](nc)
            val dist_X = DenseMatrix.zeros[Double](m, nc)
            listX.zipWithIndex.foreach{ case(el, i) =>
                val res = text2arr(el, config.delimiter, config.hasLabel)
                dist_X(::, i) := res._1
                labels(i) = res._2
            }
            val indexTuple = (index, nc)
            Array(((indexTuple, dist_X), labels)).iterator
        })
        result.count() // take action
        val features = result.map(_._1)
        features.count()
        val labels = result.flatMap(_._2)
        //labels.count()
        /**
         * Initialization method
         */
        if (config.init_method == "randomCols"){
            // Initialize W with random sample
            val W_arr = X.takeSample(false, config.rank).map( w => {
                val res = text2arr(w, config.delimiter, config.hasLabel)
                res._1
            })
            W_arr.zipWithIndex.foreach{ case(el, i) => W(::, i):= el}
        } else if (config.init_method == "kMeans"){
            // Initialize W ith K-Means with random sample
            val factor = 1000
            val fraction = 0.02d
            val rng = scala.util.Random
            val subX = X.filter(x => rng.nextFloat < fraction).map(w => {
                val res = text2arr(w, config.delimiter, config.hasLabel)
                Vectors.dense(res._1.toArray)}).cache()
            println("subX", subX.count())
//            val sampleInstances = X.takeSample(false, min(config.rank * factor, 10000)
//            ).map( w => {val res = text2arr(w, config.delimiter, config.hasLabel)
//                Vectors.dense(res._1.toArray)})
//            val sampleInstancesRDD = sc.parallelize(sampleInstances).cache()
            //val sampleInstancesRDD = X.map(w => TextConverter.text2Arr(w, config.delimiter, config.hasLabel)._1).cache()
            val model = new KMeans().setK(config.rank).setMaxIterations(100).run(subX)
            val clusters = model.clusterCenters
//            sampleInstancesRDD.unpersist()
            W := new DenseMatrix[Double](m, config.rank, clusters.flatMap(_.toArray))
        } else if (config.init_method =="Random") {
            val normal01 = breeze.stats.distributions.Gaussian(0, 1)
            W := DenseMatrix.rand(m, config.rank, normal01)
        } else {
            throw new IllegalArgumentException("Illegal initialization method.")
        }
        X.unpersist()
        result.unpersist()
        (features, labels, W)
    }

    def main(args: Array[String]): Unit = {
        // parse input
        val builder = OParser.builder[DBMDConfig]
        val parser1 = {
            import builder._
            OParser.sequence(
                programName("SparkDBMD"),
                head("SparkDBMD", "0.1"),
                opt[String]('f', "input").required().
                    action( (x, c) => c.copy(input = x)).
                    text("path/to/input file"),
                opt[Int]('n', name = "node").required().
                    action( (x, c) => c.copy(node = x)).
                    text("number of nodes to be used"),
                opt[Int]('r', name = "rank").required().
                    action( (x, c) => c.copy(rank = x)).
                    text("rank of the decomposition"),
                opt[Double]('l', "lam").required().
                    action( (x, c) => c.copy(lam = x)).
                    text("l1 penalized parameter"),
                opt[Double]("hp").required().
                    action( (x, c) => c.copy(hp = x)).
                    text("hyper parameter"),
                opt[String]('o', "output").required().
                    action( (x, c) => c.copy(output = x)).
                    text("path/to/put file"),
                opt[Int]( name = "maxIter").optional().
                    action( (x, c) => c.copy(maxIter = x)).
                    text("maximum iteration times"),
                opt[Double]('t', name = "tol").optional().
                    action( (x, c) => c.copy(tol = x)).text("tolerance"),
                opt[Unit](name = "hasLabel").
                    action( (_, c) => c.copy(hasLabel=true))
                    .text("data matrix has label indicator"),
                opt[Unit](name = "cv").
                    action( (_, c) => c.copy(cv=true))
                    .text("Run cross validation for parameter selection."),
                opt[String](name = "delimiter").optional().action((x, c) => c.copy(delimiter = x)).
                    text("Specify the delimiter of input file."),
                opt[Double](name = "alpha").optional().action((x, c) => c.copy(alpha = x)).
                    text("Dirichlet prior"),
                opt[String](name = "init_method").optional().action( (x, c) => c.copy(init_method = x)).
                    text("Initialization method: randomCols or kMeans"),
                opt[Double](name = "std").optional.action((x, c) => c.copy(std = x)).
                    text("proportion of noise"),
                opt[String](name = "algorithm").action( (x, c) => c.copy(algorithm=x)),
                opt[Unit](name = "dropConstraints").
                    action( (_, c) => c.copy(dropConstraints=true))
                    .text("data matrix has label indicator"),
                opt[Int](name = "rep").action((x, c) => c.copy(repeat = x)).text("repeat the algorithm")
            )
        }
        OParser.parse(parser1, args, DBMDConfig()) match {
            case Some(config) =>{
                val spark = SparkSession.builder().getOrCreate()
                val sc = spark.sparkContext
                sc.setLogLevel("WARN")
                println("=============================DBMD=======================\n")
                println(">>Input file: " + config.input)
                for (rep <- 1 to config.repeat) {
                    print("=====================Repeat=%d===================\n" format rep)
                    val dataset = process(sc, config)
                    var X = dataset._1
                    if (config.std > 0) {
                        println(">>Noise std = %.2f" format config.std)
                        val s = NoiseSituation.getSituation(config.std)
                        X = X.map(x => AddNoiseMat.addGaussianMixture(x, s.stdVector, s.mixtureProp)).cache()
                    }
                    val initW = dataset._3
                    val alpha = DenseVector.fill(config.rank, config.alpha)
                    val model = config.algorithm match {
                        case "ADMM" => new AdmmDBMD(alpha, config.lam, verbose = false)
                        case "AGD" => new AgdDBMD(alpha, config.lam, verbose = false)
                        case "CEASE" => new CeaseDBMD(alpha, config.lam, verbose = false)
                        case _ => throw new Exception("Not implemented algorithm. Algorithm should be ADM, AGD or CEASE")
                    }

                    val hp = (config.algorithm, config.hp)
                    if (config.cv) {
                        println("")
                        // model.cv(sc, X, initW, hp, config.tol, config.maxIter, 0)
                    } else {
                        val trueLabels = dataset._2
                        val fitRes = model.fit(sc, X, initW, config.rank, hp, config.maxIter, config.tol,
                            dropConstraints = config.dropConstraints)
                        val est_Hc = fitRes._2
                        val predLabels = model.getLabels(est_Hc)
                        val metrics = new ExternalMetrics(trueLabels, predLabels)
                        println("FMeasure = %.4f" format (metrics.fMeasure()))
                        println("Accuracy = %.4f" format (metrics.accuracy()))
                        MatUtil.csvWriter(fitRes._1, config.output.concat("W.txt"))
                    }
                }
            }
            case None =>
                println("Input arguments is not valid")
                System.exit(0)
        }
    }

}
