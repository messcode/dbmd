package com.dbmd

import breeze.linalg.{DenseMatrix, DenseVector, eigSym}
import breeze.numerics.{abs, sqrt}
import com.utils.MatUtil
import com._
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast

import scala.util.control.Breaks.{break, breakable}




class CEADBMD(sc: SparkContext, C:Int, rank: Int, alpha: DenseVector[Double], lam: Double) extends
DBMD(sc: SparkContext, C:Int, rank: Int, alpha: DenseVector[Double], lam: Double){

    def initCEA(L: idResRDD, bcastW: BcastMat): idResRDD = {
        val curL = L.map(l => {
            val (index, xc, _, hc, _, vars, del_Hc) = l
            (index, xc, bcastW.value, hc, hc * hc.t, vars, del_Hc)
        })
        curL
    }

    override def updateW(L: idResRDD, W: Mat, bcastW: BcastMat, bcastLam: Broadcast[Double],
                         hp: hyperParameter, tol: Double, maxIter: Int = 30):
    (idResRDD, Mat, BcastMat, String) = {
        var curL = initCEA(L, bcastW)
        val gamma = hp._2
        val den = curL.map(l => 1d / l._6).reduce(_ + _)
        val bcastDen = sc.broadcast(den)
        val bcastGamma = sc.broadcast(gamma)
        val bcastTol = sc.broadcast(tol)
        var (curW, curBcastW) = (W, bcastW)
        var subIter = -1
        var finalDelW = 0d
        var preW = curW
        val wTol = MatUtil.norm(W) * tol
        breakable {
            for (nIter <- 0 to maxIter) {
                // compute (weighted) gradient
                val gradF = curL.map(l => (curBcastW.value * l._4 - l._2) * l._4.t *:* (1d / l._6 / bcastDen.value)).reduce(_+_)
                val bcastGradF = sc.broadcast(gradF)
                val bcastC = sc.broadcast(C)
                // update on nodes
                curL = curL.map(l=> Mappers.fista(l, curBcastW.value, bcastLam.value / bcastC.value,
                    bcastGamma.value, bcastGradF.value, tol=bcastTol.value))
                // averaging on central processor
                curW = curL.map(l => l._3 *:* (1d / l._6 /  bcastDen.value)).reduce(_ + _)
                // Check relative change of W
                val delW = MatUtil.norm(preW - curW) / sqrt(preW.size * 1d)
                preW = curW
                curBcastW = sc.broadcast(curW)
                subIter = nIter
                finalDelW = delW
                if (delW < wTol && nIter >= 1) {
                    break()
                }
            }
        }
        val msg = "InnerLoop=%d delW=%.8f avg(curW)=%.4f" format(subIter + 1, finalDelW,
            MatUtil.normAvg(curW))
        (curL, curW, curBcastW, msg)

    }
}
