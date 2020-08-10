package com.dbmd

import breeze.linalg.{DenseMatrix, DenseVector, eigSym, inv}
import breeze.numerics.sqrt
import com.utils.MatUtil
import com.{BcastMat, Mat, Vec, hyperParameter, indexedMat}
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

import scala.util.control.Breaks.{break, breakable}

class AgdDBMD(alpha: Vec, lam: Double, name: String = "AGD-DBMD", verbose: Boolean = false) extends
    DistributedBMD(alpha: Vec, lam: Double, name: String, verbose) {
    override def updateWAndWc(sc: SparkContext, numNodes: Int, resRDD: RDD[ResHolder], W: Mat, hp: hyperParameter, tol: Double = 1e-4,
                              maxIter: Int = 35, latency: Long = 0): Mat = {
        resRDD.foreach(res => res.Uc = res.Hc * res.Hc.t)
        val Lp = eigSym(resRDD.map(res => res.Uc).reduce(_ + _)).eigenvalues(-1)
        val step = 1d / Lp
        var curY = W
        var preW = W
        var curW = W
        var curBcastY = sc.broadcast(curW)
        var t = 1d
        val wTol = MatUtil.norm(W) * tol
        breakable {
            for (nIter <- 1 to maxIter) {
                // compute gradient
                Thread.sleep(latency)
                val grad = resRDD.map(res => res.Wc * res.Uc - res.Xc * res.Hc.t).reduce(_ + _)
                curW = curY - grad * step
                curW = MatUtil.softh(curW, this.lam * step)
                val num = t - 1
                t = .5d + sqrt(1 + 4 * t * t)
                curY = curW + (curW - preW) *:* (num / t)
                preW = curW
                // broadcast Y
                curBcastY = sc.broadcast(curY)
                resRDD.foreach(res => res.Wc = curBcastY.value)
                val delW = MatUtil.norm(preW - curW)
                if (delW < wTol &&  nIter >= 29) break()
            }
        }
        curW
    }
}


class AdmmDBMD(alpha: Vec, lam: Double, name: String = "ADMM-DBMD", verbose: Boolean = false) extends
    DistributedBMD(alpha, lam, name, verbose) {
    override def updateWAndWc(sc: SparkContext, numNodes: Int, resRDD: RDD[ResHolder], W: Mat, hp: hyperParameter, tol: Double = 1e-4,
                              maxIter: Int = 10, latency: Long = 0): Mat = {
        // set Uc = 0
        resRDD.foreach(res => res.Uc = res.Uc *:* 0d)
        var curW = W
        var preW = W
        val rho = hp._2
        val den = resRDD.map(res => 1d / res.vars).reduce(_ + _)
        val wTol = MatUtil.norm(W) * tol
        breakable {
            for (nIter <- 1 to maxIter) {
                // update Wc
                resRDD.foreach(res => res.Wc = ((res.Xc * res.Hc.t + res.Uc) /:/ rho + W) *
                    inv(DenseMatrix.eye[Double](res.Hc.rows) + res.Hc * res.Hc.t /:/ rho))
                // update W
                Thread.sleep(latency)
                curW = resRDD.map(res => (res.Wc - res.Uc /:/ rho) *:* (1d / res.vars / den)).reduce(_ + _)
                curW = MatUtil.softh(curW, this.lam / numNodes.toDouble / rho)
                // check the relative change of W
                val delW = MatUtil.norm(preW - curW)
                preW = curW
                val curBcastW = sc.broadcast(curW)
                // update Uc
                resRDD.foreach(res => res.Uc = res.Uc + res.Wc - curBcastW.value)
                if (delW < wTol &&  nIter >= 5) break()
            }
        }
        curW
    }
}


class CeaseDBMD(alpha: Vec, lam: Double, name: String = "CEASE-DBMD", verbose: Boolean = false) extends
    DistributedBMD(alpha, lam, name, verbose) {

    override def updateWAndWc(sc: SparkContext, numNode: Int, resRDD: RDD[ResHolder], W: Mat, hp: hyperParameter, tol: Double = 1e-4,
                              maxIter: Int = 5, latency: Long = 0): Mat = {

        val gamma = hp._2
        val den = resRDD.map(res => 1d / res.vars).reduce(_ + _)
        val bcastDen = sc.broadcast(den)
        val bcastGamma = sc.broadcast(gamma)
        val bcastLam = sc.broadcast(this.lam / numNode.toDouble)
        var curBcastW = sc.broadcast(W)
        var preW = W
        val wTol = MatUtil.norm(W) * tol
        breakable {
            for (nIter <- 1 to maxIter) {
                // compute weighted gradient
                Thread.sleep(latency)
                val gradF = resRDD.map(res => (curBcastW.value * res.Hc - res.Xc) *
                    res.Hc.t *:* (1d / res.vars / bcastDen.value )).reduce(_ + _)
                val bcastGradF = sc.broadcast(gradF)
                // update on nodes
                Thread.sleep(latency)
                resRDD.foreach(res => MapperCeaseWc.update(res, curBcastW.value, bcastGradF.value,
                    bcastGamma.value, bcastLam.value))
                // averaging on the central processor
                val curW =  resRDD.map(res => res.Wc *:* (1d / res.vars /  bcastDen.value)).reduce(_ + _)
                curBcastW = sc.broadcast(curW)
                val delW = MatUtil.norm(preW - curW)
                preW = curW
                if (delW < wTol &&  nIter >= 5) break()
            }
        }
        W
    }
}