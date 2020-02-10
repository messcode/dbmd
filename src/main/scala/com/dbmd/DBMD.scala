package com.dbmd

import breeze.linalg.{Axis, DenseMatrix, DenseVector, argmax, max, min, norm, sum}
import breeze.numerics.{abs, sqrt}
import breeze.stats.distributions.Dirichlet
import com._
import com.utils.MatUtil
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.log4j.{Level, Logger}
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter


import scala.util.control.Breaks.{break, breakable}
import com.typesafe.scalalogging._
/**
 * Base class for distributed Matrix Decomposition
 * @param sc SparkContext
 * @param C  number of
 * @param rank decomposition method
 * @param alpha Dirichlet prior on the d matrix H
 * @param lam Laplace prior on the basis matrix W
 */
abstract class DBMD (sc: SparkContext, C:Int,
                     rank: Int, alpha: DenseVector[Double], lam: Double)
{
    /**
     *  hold the indexed round results for each round of computation
     *      AGD:   (index, Xc, Wc, Hc, Hc * Hc', vars \delta Hc)
     *      ADMM:  (index, Xc, Wc, Hc, Uc, vars, \delta Hc)
     *      CEASE: (index, Xc, Wc, Hc, 0, vars \delta Hc)
     */
    val logger = Logger.getLogger("DBMD")
    logger.setLevel(Level.DEBUG)
    logger.info("Distributed Bayesian Matrix Decomposition")
    val bcastLam: Broadcast[Double] = sc.broadcast(lam)
    val bcastPhi: Broadcast[(Int, Int)] = sc.broadcast((1, 1))
    val bcastC: Broadcast[Int] = sc.broadcast(C)
    val bcastAlpha: Broadcast[DenseVector[Double]] = sc.broadcast(alpha)

    def initialize(sc: SparkContext, X: indexedMat, initW: Mat):
    (idResRDD, Mat, BcastMat) =
    {
        val bcastW = sc.broadcast(initW)
        val bcastRank = sc.broadcast(rank)
        val bcastAlpha = sc.broadcast(alpha)
        val L = X.map(x => {
            val rank = bcastRank.value
            val alpha = bcastAlpha.value
            val index = x._1
            val Xc = x._2
            val Wc: DenseMatrix[Double] = bcastW.value
            val n = Xc.cols
            val dirchilet = Dirichlet(alpha.toArray).sample(n)
            val Hc: DenseMatrix[Double] = DenseMatrix.zeros[Double](rank, n)
            for (i <- dirchilet.indices) {
                Hc(::, i) := dirchilet(i)
            }
            val Uc = DenseMatrix.zeros[Double](initW.rows, initW.cols)
            val variance = 1d
            val del_Hc = 0d
            (index, Xc, Wc, Hc, Uc, variance, del_Hc)
        }).cache()
        L.count()
        X.unpersist()
        (L, initW, bcastW)
    }

    def updateW(L: idResRDD, W: DenseMatrix[Double], bcastW: Broadcast[DenseMatrix[Double]], bcastLam: Broadcast[Double],
                hp: hyperParameter, tol: Double, maxIter: Int=30) :
    (idResRDD, DenseMatrix[Double], Broadcast[DenseMatrix[Double]], String)

    def updateHc(L: idResRDD, bro_W: Broadcast[DenseMatrix[Double]], bro_alpha: Broadcast[DenseVector[Double]]): idResRDD =
    {
        val curL = L.map(l => Mappers.updateHc(l, bro_alpha.value)).cache()
        curL.count()
        L.unpersist()
        curL
    }

    def calAvgResidual(L: idResRDD): Double ={
        val avgRes = L.map(l => sum(abs(l._2 - l._3 * l._4)) / l._2.size.toDouble).mean()
        avgRes
    }


    def calObjVal(L: idResRDD, W: DenseMatrix[Double], bcastW: Broadcast[DenseMatrix[Double]]): Double = {
        val avgObjVal = .5d * L.map(l => MatUtil.squareNorm(l._2 - bcastW.value * l._4)).mean() + sum(abs(W)) * lam / C
        avgObjVal / W.size
    }

    def calAvgSparseness(mL: idResRDD): Double = {
        val avg_sp = mL.map(l => MatUtil.sparseness(l._4)).mean()
        avg_sp
    }

    def getMaxDelHc(L: idResRDD): Double = L.map(_._7).max()

    /**
     *
     * @param sc
     * @param X indexedMat
     * @param initW
     * @param hp hyper parameter
     * @param tol tolerance
     * @param maxIter maximum number of iteration
     * @return
     */
    def fit(sc: SparkContext, X: indexedMat, initW: DenseMatrix[Double], hp: (String, Double),
            tol: Double, maxIter: Int, wTol: Double = 1e-2, minIter: Int = 10):
    (RDD[DenseMatrix[Double]],  DenseMatrix[Double]) =
    {
        var (curL, curW, curBcastW) = initialize(sc, X, initW)
        val delHc = new Array[Double](maxIter)
        val objVals = new Array[Double](maxIter)
        var totalIter = maxIter
        logger.info("Running %s-DBMD" format hp._1)
        val st = System.currentTimeMillis()
        breakable {
           for (nIter <- 0 until maxIter) {
               val sigmas = new DenseVector[Double](curL.map(_._6).collect())
               curL = updateHc(curL, curBcastW, bcastAlpha)
               logger.debug("Iter=%d || Procedure=Update Hc||Avg(W)=%.4f" format(nIter, MatUtil.normAvg(curW)))
               val LWW = updateW(curL, curW, curBcastW, bcastLam, hp, wTol)
               curL = LWW._1
               curW = LWW._2
               curBcastW = LWW._3
               logger.debug("Iter=%d || Procedure=Update W||Avg(W)=%.4f" format(nIter, MatUtil.normAvg(curW)))
               logger.debug("Iter=%d || Procedure=Update W||Msg=%s" format(nIter, LWW._4))
               delHc(nIter) = getMaxDelHc(curL)
               objVals(nIter) = calObjVal(curL, curW, curBcastW)
               logger.debug("Iter=%d||Summary||Sigma:min=%.4f mean=%.4f max=%.4f " format(nIter, min(sigmas),
                   sum(sigmas) / sigmas.length, max(sigmas)))
               logger.debug("Iter=%d||Summary||Avg(res.)=%.4f" format(nIter, calAvgResidual(curL)))
               logger.debug("Iter=%d||Summary||delHc=%.4f" format(nIter, delHc(nIter)))
               val dt = DateTimeFormatter.ofPattern("MM-dd HH:mm:ss").format(LocalDateTime.now)
               val msg = "Iter=%d||Summary||objVal=%.4f" format(nIter, objVals(nIter))
               logger.info(msg)
               println(dt + "||" + msg)
               // stopping criteria
               if (nIter > minIter - 1) {
                    val delObjval =  abs(objVals(nIter-1) - objVals(nIter)) / abs(objVals(nIter))
                    if (delHc(nIter) < 1e-3 && delObjval < tol) {
                        totalIter = nIter
                        break()
                    }
               }
           }
        }
        val duration = (System.currentTimeMillis() - st) / 1000f
        println("#totalIter=%d Time=%.4f" format(totalIter, duration))
        val Hc = curL.map(_._4).cache()
        (Hc, curW)
    }

    def predict(sc: SparkContext, W: DenseMatrix[Double], hoX: indexedMat): idResRDD = {
        val bcastW = sc.broadcast(W)
        val bcastAlpha: Broadcast[DenseVector[Double]] = sc.broadcast(alpha)
        val bcastRank = sc.broadcast(rank)
        val L = hoX.map(x => {
            val r = bcastRank.value
            val alpha = bcastAlpha.value
            val index = x._1
            val Xc = x._2
            val Wc: DenseMatrix[Double] = bcastW.value
            val n = Xc.cols
            val dirichlet = Dirichlet(alpha.toArray).sample(n)
            val Hc: DenseMatrix[Double] = DenseMatrix.zeros[Double](r, n)
            for (i <- dirichlet.indices) {
                Hc(::, i) := dirichlet(i)
            }
            val Uw = DenseMatrix.zeros[Double](r, n)
            val variance = 1d
            val del_Hc = 0d
            (index, Xc, Wc, Hc, Uw, variance, del_Hc)
        }).cache()
        L.count()
        hoX.unpersist()
        updateHc(L, bcastW, bcastAlpha)
    }

    def partitionMat(X: indexedMat, ho:Int): (indexedMat, indexedMat) = {
        val trainX = X.filter(_._1._1 != ho)
        val hoX = X.filter(_._1._1 == ho)
        (trainX, hoX)
    }


    def cv(sc:SparkContext, X: indexedMat, initW : DenseMatrix[Double], hp: hyperParameter,
           tol: Double, maxIter: Int, hoIndex: Int): Double ={
        assert(hoIndex > -1 && hoIndex < C)
        val (trainX, hoX) = partitionMat(X, hoIndex)
        val fitRes = fit(sc, trainX, initW, hp, tol, maxIter)
        val predRes = predict(sc, fitRes._2, hoX)
        val broW = sc.broadcast[DenseMatrix[Double]](fitRes._2)
        val reconErr = predRes.map(l => MatUtil.normAvg(l._2 - broW.value * l._4)).mean()
        println("Reconstruction error on holdout data = %.4f" format(reconErr))
        reconErr
    }

    /**
     * Return the indexes of maximum  as labels .
     * Note that the number of class may be smaller than the given rank.
     * */
    def getLabels(Hc: RDD[DenseMatrix[Double]]): RDD[Int] ={
        val predLabel = Hc.flatMap(argmax(_, Axis._0).t.toArray)
        predLabel.map(_ + 1)
    }
}

