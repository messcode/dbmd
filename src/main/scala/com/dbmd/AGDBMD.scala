package com.dbmd

import breeze.linalg.{DenseMatrix, DenseVector, eigSym}
import breeze.numerics.sqrt
import com.utils.MatUtil
import com.{hyperParameter, idResRDD}
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast

import scala.util.control.Breaks.{break, breakable}


class AGDBMD (sc: SparkContext, C:Int, rank: Int, alpha: DenseVector[Double], lam: Double) extends
    DBMD(sc: SparkContext, C:Int, rank: Int, alpha: DenseVector[Double], lam: Double) {

    def initAGD(L: idResRDD, bcastW: Broadcast[DenseMatrix[Double]]): idResRDD = {
        val curL = L.map(l => {
            val (index, xc, _, hc, _, vars, delHc) = l
            (index, xc, bcastW.value, hc, hc * hc.t, vars, delHc)
        })
        curL
    }

    override def updateW(L: idResRDD, W: DenseMatrix[Double], bcastW: Broadcast[DenseMatrix[Double]], bcastLam: Broadcast[Double],
                         hp: hyperParameter, tol: Double, maxIter: Int = 50):
    (idResRDD, DenseMatrix[Double], Broadcast[DenseMatrix[Double]], String) = {
        var (curL, curW, curBcastW) = (L, W, bcastW)
        curL = initAGD(curL, curBcastW)
        val Lp = eigSym(curL.map(l => l._5).reduce(_ + _)).eigenvalues(-1)
        var curY = curW
        var preW = curW
        var subIter = 1
        var t = 1d
        var finalDelW = 0d
        val wTol = MatUtil.norm(W) * tol
        breakable {
            for (nIter <- 0 until maxIter) {
                // compute grad
                var grad = curL.map(l => {
                    val (_, xc, _, hc, hht, _, _) = l
                    curBcastW.value * hht - xc * hc.t
                }).reduce(_ + _)
                curW = curY - grad /:/ Lp
                curW = MatUtil.softh(curW, lam / Lp)
                curBcastW = sc.broadcast(curW)
                val delW = MatUtil.norm(preW - curW) / sqrt(preW.size * 1d)
                val num = t - 1
                t = .5d + sqrt(1 + 4 * t * t)
                curY = curW + (curW - preW) *:* (num / t)
                preW = curW
                subIter = nIter
                finalDelW = delW
                if (delW < wTol && nIter >= 30){
                    break()
                }
            }
        }
        val msg = "InnerLoop=%d  delW=%.8f Lp=%.4f" format(subIter, finalDelW, Lp)
        (curL, curW, curBcastW, msg)
    }

}
