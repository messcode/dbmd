package com.dbmd

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.{abs, sqrt}
import com.{hyperParameter, idResRDD}
import com.utils.MatUtil
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast

import scala.util.control.Breaks.{break, breakable}

class ADMDBMD(sc: SparkContext, C:Int, rank: Int, alpha: DenseVector[Double], lam: Double) extends
    DBMD(sc: SparkContext, C:Int, rank: Int, alpha: DenseVector[Double], lam: Double) {

    def initADM(L: idResRDD, bcastW: Broadcast[DenseMatrix[Double]]): idResRDD = {
        val curL = L.map(l => {
            val (index, xc, _, hc, uc, vars, del_Hc) = l
            (index, xc, bcastW.value, hc, uc *:* 0d, vars, del_Hc)
        })
        curL
    }

    override def updateW(L: idResRDD, W: DenseMatrix[Double], bcastW: Broadcast[DenseMatrix[Double]], bcastLam: Broadcast[Double],
                         hp: hyperParameter, tol: Double, maxIter: Int = 10):
    (idResRDD, DenseMatrix[Double], Broadcast[DenseMatrix[Double]], String) = {
        var (curL, curW, curBcastW) = (L, W, bcastW)
        curL = initADM(L, bcastW)
        var preW = curW
        val rho = hp._2
        val den = curL.map(l => 1d / l._6).reduce(_ + _)
        val bcastDen = sc.broadcast(den)
        val bcastRho = sc.broadcast(rho)
        var subIter = -1
        var finalDelW = 0d
        val wTol = MatUtil.norm(W) * tol
        breakable {
            for (nIter <- 0 until maxIter) {
                // Update Wc
                curL = curL.map(l => Mappers.updateWc(l, curBcastW.value, bcastRho.value))
                curL.count()
                // Update W
                curW = curL.map(l => (l._3 - l._5 /:/ bcastRho.value) *:* (1d / l._6 / bcastDen.value)).reduce(_ + _)
                curW = MatUtil.softh(curW, lam / (C.toDouble * rho))
                // Check relative change of W
                val delW = MatUtil.norm(preW - curW)
                preW = curW
                curBcastW = sc.broadcast(curW)
                // Update Uc
                curL = curL.map(l => Mappers.updateUc(l, curBcastW.value))
                subIter = nIter
                finalDelW = delW
                if (delW < wTol && nIter >= 1) {
                    break()
                }
            }
        }
        val viol = curL.map(l => MatUtil.normAvg(curBcastW.value - l._3)).mean()
        val msg = "InnerLoop=%d Viol=%.4f delW=%.8f" format(subIter + 1, viol, finalDelW)
        (curL, curW, curBcastW, msg)
    }
}