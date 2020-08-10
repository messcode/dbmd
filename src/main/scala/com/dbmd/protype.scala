package com.dbmd

import breeze.linalg.{Axis, DenseMatrix, argmax, sum}
import breeze.numerics.abs
import breeze.stats.distributions.Dirichlet
import com.utils.MatUtil
import com.{BcastMat, Mat, Vec, hyperParameter, indexedMat}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import scala.util.control.Breaks.breakable
/**
 *  Serializable class to store the parameters.
 */
class ResHolder(ind: (Int, Int), xc: Mat, wc: Mat, hc: Mat, uc: Mat,
                         sigmas: Double, dhc: Double) extends  Serializable {
    val index = ind
    val Xc = xc
    var Hc = hc
    var Wc = wc
    var Uc = uc
    var vars = sigmas
    var dHc = dhc

    def setWc(newWc: Mat): ResHolder = {
        Wc = newWc
        this
    }

    def setHc(newHc: Mat): ResHolder = {
        Hc = newHc
        this
    }

    def setUc(newUc: Mat): ResHolder = {
        Uc = newUc
        this
    }

    def setVars(newVars: Double): ResHolder = {
        vars = newVars
        this
    }

    def setDeltaHc(newDeltaHc: Double): ResHolder = {
        dHc = newDeltaHc
        this
    }
}

abstract class DistributedBMD(alpha: Vec, lam: Double, name: String, verbose: Boolean=true) {

    val logger = Logger.getLogger(name)


    if (verbose) {
        logger.setLevel(Level.DEBUG)
    } else {
        logger.setLevel(Level.INFO)
    }

    logger.info("Distributed Bayesian Matrix Decomposition")

    def calObjVal(n: Int, resRDD: RDD[ResHolder], W:Mat, bcastW: BcastMat): Double = {
        val objVal = .5d * resRDD.map(res => MatUtil.norm(res.Xc - res.Wc * res.Hc) / res.Wc.rows.toDouble).sum()
        + this.lam * sum(abs(W)) / W.rows.toDouble
        objVal
    }

    def initialize(sc: SparkContext, r: Int, X: indexedMat, bcastW: BcastMat): RDD[ResHolder] = {
        val bcastR = sc.broadcast(r)
        val bcastAlpha = sc.broadcast(alpha)
        val resRDD = X.map(x => {
            val n = x._2.cols
            val dirchilet = Dirichlet(bcastAlpha.value.toArray).sample(n)
            val Hc = DenseMatrix.zeros[Double](bcastR.value, n)
            for (i <- dirchilet.indices) {
                Hc(::, i) := dirchilet(i)
            }
            val Uc = DenseMatrix.zeros[Double](bcastW.value.rows, bcastW.value.cols)
            new ResHolder(x._1, x._2, bcastW.value, Hc, Uc, 1d, 0d)
        })
        resRDD
    }

    def updateWAndWc(sc: SparkContext, numNodes: Int, resRDD: RDD[ResHolder], W: Mat, hp: hyperParameter, tol: Double = 1e-4,
                     maxIter: Int = 35, latency: Long = 0L): Mat

    /**
     * Fit the DistributedBMD model.
     * @param sc: SparkContext
     * @param X: RDD of indexes and matrix tuples RDD[((Int, Int), Mat)]
     * @param W0: initialized basis matrix
     */
    def fit(sc: SparkContext, X: indexedMat, W0: Mat, r: Int, hp: (String, Double),
        maxIter: Int = 10, tol: Double, dropConstraints: Boolean = false, latency: Long = 0L): (Mat, RDD[Mat]) = {
        val numNodes = X.count().toInt
        var W = W0
        val bcastW = sc.broadcast(W0)
        val bcastAlpha = sc.broadcast(alpha)
        val resRDD = initialize(sc, r, X, bcastW).cache() // cache to memory
        var totalIter = 0
        X.unpersist()
        logger.info("Running %s" format this.name)
        if (dropConstraints) logger.info("Drop constraints of Hc. " +
            "Update with Projected gradient algorithm.")
        val st = System.currentTimeMillis()
        breakable {
            for (nIter <-1 to maxIter) {
                //Update H0
                if (dropConstraints) {
                    resRDD.foreach(res => MapperHc.update(res))
                    W = resRDD.map(res => res.Wc).reduce(_ + _) /:/ numNodes.toDouble
                } else {
                    resRDD.foreach(res => MapperHcWithConstraints.update(res, bcastAlpha.value))
                }
                logger.debug("Iter=%d || Procedure=Update H||objVal=%.4f"
                    format(nIter, calObjVal(numNodes.toInt, resRDD, W, bcastW)))
                // Update W and Wc
                W = updateWAndWc(sc, numNodes, resRDD, W, hp, latency=latency)
                logger.info("Iter=%d || Procedure=Update W||objVal=%.4f"
                    format(nIter, calObjVal(numNodes.toInt, resRDD, W, bcastW)))
                totalIter = nIter
            }
        }
        val duration = (System.currentTimeMillis() - st) / 1000f
        println("#totalIter=%d Time=%.4f" format(totalIter, duration))
        (W, resRDD.map(_.Hc))
    }

    def getLabels(Hc: RDD[Mat]): RDD[Int] ={
        val predLabel = Hc.flatMap(argmax(_, Axis._0).t.toArray)
        predLabel.map(_ + 1)
    }
}

