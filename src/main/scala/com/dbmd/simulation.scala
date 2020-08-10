package com.dbmd

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.Dirichlet
import com.{Mat, indexedMat}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import scopt.OParser

case class SimulationConfig (
                           numNodes: Int = 10,
                           numPerInstances: Int = 10000,
                           numFeatures: Int = 100,
                           dist: String = "Dirichlet",
                           distParam: Double = 0.1d,
                           latency: Long = 0,
                           rank: Int = 2,
                           proportion: Double =  -1d, //
                           scale: Double = 2d,
                           output: String = "simulation.log",
                           repeat: Int = 1
                           ) {
    require(proportion<0 || (proportion > 0d && proportion < 1), "proportion parameter should be in range (0, 1)")
    require(dist == "Dirichlet" || dist == "Bernoulli",
        "dist parameter should be either `Dirichlet` or `Bernoulli`")
    require(scale > 0, "scale parameter shoulde be positive")
}


object SimulationRunner {
    /**
     *
     * @param sc     Spark context
     * @param config SimulationConfig
     * @return (W, RDD of Hc, RDD of Xc)
     */
    def genSimData(sc: SparkContext, config: SimulationConfig): (Mat, RDD[Mat], indexedMat) = {
        val numInstancesArray = if (config.proportion < 0) {
            Array.fill[Int](config.numNodes)(config.numPerInstances)
        } else {
            val numInstances = config.numPerInstances * config.numNodes
            val restNumInstances = math.ceil(numInstances * (1 - config.proportion) / (config.numNodes - 1).toDouble).toInt
            val numInstancesArray = Array.fill[Int](config.numNodes)(restNumInstances)
            numInstancesArray(0) = numInstances - restNumInstances * (config.numNodes - 1)
            numInstancesArray
        }
        val numInstancesRDD = sc.parallelize(numInstancesArray).repartition(config.numNodes)
        val bcastR = sc.broadcast(config.rank)
        val normal = breeze.stats.distributions.Gaussian(0, config.scale)
        val W = DenseMatrix.rand(config.numFeatures, config.rank, normal)
        val bcastW = sc.broadcast(W)
        val HcRDD = if (config.dist == "Dirichlet") {
            val alpha = DenseVector.fill(config.rank, config.distParam)
            val bcastAlpha = sc.broadcast(alpha)
            numInstancesRDD.map { n => {
                val dirchilet = Dirichlet(bcastAlpha.value.toArray).sample(n)
                val Hc = DenseMatrix.zeros[Double](bcastR.value, n)
                for (i <- dirchilet.indices) {
                    Hc(::, i) := dirchilet(i)
                }
                Hc
            }
            }
        } else {
            throw new Exception("Bernoulli distribution not implemented.")
        }

        val XcRDD = HcRDD.zipWithIndex.map { case (hc, ind) => {
            val Xc = bcastW.value * hc
            ((ind.toInt, hc.cols), Xc)
        }
        }
        (W, HcRDD, XcRDD)
    }


    def main(args: Array[String]): Unit = {
        val builder = OParser.builder[SimulationConfig]
        val parser1 = {
            import builder._
            OParser.sequence(
                programName("SimulationRunner"),
                head("SimulationRunner", "0.1"),
                opt[Int]('n', "numNodes").required().
                    action((x, c) => c.copy(numNodes = x)).text("number of nodes"),
                opt[Int]('m', "numPerInstances").
                    action((x, c) => c.copy(numPerInstances = x)).text("number of instances per node machines"),
                opt[String]("dist").action((x, c) => c.copy(dist = x)).
                    text("Distribution of coefficient matrix, should be either `Dirichlet` or `Bernoulli`"),
                opt[Int]('p', "numFeatures").action((x, c) => c.copy(numFeatures = x)).text("number of features"),
                opt[Double]("distParam").action((x, c) => c.copy(distParam = x)).text("Distribution parameters."),
                opt[Long]("latency").action((x, c) => c.copy(latency = x)).text("communication latency (milliseconds) "),
                opt[Int]('r', "rank").action((x, c) => c.copy(rank = x)).text("rank of decomposition"),
                opt[Double]("proportion").action((x, c) => c.copy(proportion = x)).
                    text("proportion of the first machine, instances are evenly distributed when this parameter < 0"),
                opt[Double]("scale").action((x, c) => c.copy(scale = x)).text("scale of the basis matrix `W`"),
                opt[String]('o', "output").action( (x, c) => c.copy(output = x)).text("path/to/output/log/file"),
                opt[Int]("rep").action( (x, c) => c.copy(repeat = x)).text("Repeat algorithm")
            )
        }
        OParser.parse(parser1, args, SimulationConfig()) match {
            case Some(config) => {
                val REPLICATE = 2
                val lam = 1e-3 * config.numPerInstances
                val spark = SparkSession.builder().getOrCreate()
                val sc = spark.sparkContext
                sc.setLogLevel("WARN")
                println("\nSimulation parameters>>numNodes = %d, proportion = %f, latency = %d, numInstances = %d"
                    format(config.numNodes, config.proportion, config.latency, config.numNodes * config.numPerInstances))
                val dataset = genSimData(sc, config)
                val X = dataset._3
                val alpha = DenseVector.fill(config.rank, 0.1d)
                X.foreach(x => println(x._2.rows, x._2.cols)) // print x size for validation
                val agdModel = new AgdDBMD(alpha, lam, verbose = false)
                val admmModel = new AdmmDBMD(alpha, lam, verbose = false)
                val ceaseModel = new CeaseDBMD(alpha, lam , verbose = false)
                val normal = breeze.stats.distributions.Gaussian(0, config.scale)
                for (rep <- 1 to config.repeat) {
                    println("=============================Repeat=%d=============================" format rep)
                    val initW = DenseMatrix.rand(config.numFeatures, config.rank, normal)
                    agdModel.fit(sc, X, initW, config.rank, ("agd", 0d), 5, 1e-3, latency = config.latency)
                    admmModel.fit(sc, X, initW, config.rank, ("rho", 300d), 5, 1e-3, latency = config.latency)
                    ceaseModel.fit(sc, X, initW, config.rank, ("gamma", 0.0001d), 5, 1e-3, latency = config.latency)
                }
            }
            case _ => {
                println("Invalid input arguments.")
                System.exit(0)
            }
        }

    }
}