/*
 * In-place update functions. All updating algorithm return
 */
package com.dbmd
import breeze.linalg.{*, DenseMatrix, DenseVector, all, diag, max, min, norm, sum}
import breeze.numerics.{abs, sqrt}
import breeze.stats.distributions.Rand
import com.utils.MatUtil
import com.{Mat, Vec, idRes}

import scala.util.control.Breaks.{break, breakable}

/**
 *  Object for updating Hc with constraints that the column sum of Hc equals 1.
 */
object MapperHcWithConstraints {
    def _kkt_res(v: Vec, rank: Int, a: Vec, b: Vec, Q: Mat):
    Vec = {
        val F_a = DenseVector.zeros[Double](2 * rank + 1)
        F_a(0 until rank) := (Q * v(0 until rank)).toDenseVector - DenseVector.ones[Double](rank) *:* v(rank) - v(rank + 1 to -1) - b
        F_a(rank) = sum(v(0 until rank)) - 1
        F_a(rank + 1 to -1) := v(0 until rank) *:* v(rank + 1 to -1) - a
        F_a
    }

    def _optimize_hj(alpha: Vec, b: Vec, hj: Vec, J: Mat, Q: Mat): Vec = {
        val qp_maxiter = 50
        val rank = alpha.length
        val s = DenseVector.rand(rank, Rand.uniform)
        val rho = 0.995 // step shrinkage
        val tol = 0.001
        val eta = 0.3d // alpha shrinkage
        val Jk = J.copy
        val st = 2d
        var alpha_ = DenseVector.ones[Double](rank) * st
        Jk(rank + 1 to -1, rank + 1 to -1) := diag(hj)
        Jk(rank + 1 to -1, 0 until rank) := diag(s)
        val vk = DenseVector.zeros[Double](2 * rank + 1)
        vk(0 until rank) := hj
        vk(rank) = 1
        vk(rank + 1 to -1) := s
        breakable{
            for (n_iter <- 0 to qp_maxiter){
                val Fa = _kkt_res(vk, rank, alpha_, b, Q)
                // shrinkage aplha_
                alpha_ := max(alpha_ * eta, alpha)
                if (sum(abs(Fa)) < tol)
                    break()
                val d = Jk \ -Fa
                if (d.length == 0)
                    break()
                val step = vk /:/ d
                var step_length = 1d
                step(rank) = -1d
                if(all(step <:< 0d))
                    break()
                else{
                    val max_step = min(step.toArray.filter(_>0d))
                    step_length = min(1d, max_step*rho)
                }
                vk := vk + step_length * d
                Jk(rank+1 to -1, rank+1 to -1) := diag(vk(0 until rank))
                Jk(rank+1 to -1, 0 until rank) := diag(vk(rank+1 to -1))
                // Shrink alpha

            }
        }
        vk(0 until rank)
    }

    def update(res: ResHolder, alpha: DenseVector[Double]): Unit = {
        val rank = res.Hc.rows
        val eps = 1e-5
        val o_hc = res.Hc.copy
        def _J_Q() = {
            val Q = res.Wc.t * res.Wc
            val J = DenseMatrix.zeros[Double](2 * rank + 1, 2 * rank + 1)
            J(0 until rank, 0 until rank) := Q
            J(0 until rank, rank) := -1d
            J(0 until rank, (rank + 1) to -1) := -DenseMatrix.eye[Double](rank)
            J(rank, 0 until rank) := 1d
            (J, Q)
        }
        val (mJ, mQ) = _J_Q()
        val Bi = res.Wc.t * res.Xc
        for (idx <- 0 until res.Xc.cols){
            res.Hc(0 to -1, idx) := _optimize_hj(alpha, Bi(0 to -1, idx), res.Hc(0 to -1, idx), mJ, mQ)
        }
        res.Hc := max(res.Hc, eps)
        res.dHc = if (res.dHc != 0) norm((o_hc - res.Hc).flatten()) / norm(o_hc.flatten()) else norm(o_hc.flatten())
        res.vars = (MatUtil.squareNorm(res.Xc - res.Wc * res.Hc) + 1.0) / (4.0 + res.Xc.size)
    }
}


object MapperHc {
    def update(res: ResHolder): Unit ={
        val tol = 1e-3
        var Y = res.Hc
        var H = res.Hc
        var preH = res.Hc
        var t = 1d
        val Lp =  MatUtil.maxSymEigval(res.Wc.t * res.Wc)
        val maxIter = 30
        val WtW = res.Wc.t * res.Wc
        val WtX = res.Wc.t * res.Xc
        val hTol = MatUtil.norm(WtW * H - WtX) * tol
        breakable {
            for (nIter <- 0 until maxIter) {
                val grad = (WtW * H - WtX)
                H = Y -  grad /:/ Lp
                H(H <:< 0d) := 0d
                val numerator = t - 1
                t = .5 + sqrt(1 + 4 * t * t)
                Y  = H + (H - preH) * (numerator / t)
                preH = H
                if (MatUtil.norm(grad) < hTol) break()
            }
        }
        // normalization
        val s = sum(H(*, ::)) *:* (H.rows.toDouble / H.cols.toDouble)
        val is = s.map(x => 1d / x)
        val S = diag(s)
        val iS = diag(is)
        res.Hc = iS * H
        res.Wc = res.Wc * S
    }
}

object MapperCeaseWc {
    /**
     * Update Wc using FISTA algorithm.
     * @param gradF: global gradient
     * @param gamma: hyper parameter of CEASE
     */
    def update(res: ResHolder, W: Mat, gradF: Mat, gamma: Double, lam: Double, tol: Double = 1e-3,
               maxIter: Int = 20): Unit = {
        val HHt = res.Hc * res.Hc.t
        val XHc = res.Xc * res.Hc.t
        val gradf = (W * HHt - XHc)  - gradF
        val Lp = MatUtil.maxSymEigval(HHt + DenseMatrix.eye[Double](res.Hc.rows) *:* gamma)
        var Wc = W
        var preWc = W
        var Y = W
        var t = 1d
        val normWc = MatUtil.norm(W)
        breakable {
            for (nIter <- 0 until maxIter) {
                val grad = Y * HHt  - XHc + (Y - W) *:* gamma - gradf
                Wc = Y - grad /:/ Lp
                Wc = MatUtil.softh(Wc, lam / Lp)
                val numerator = t - 1
                t = .5 + sqrt(1 + 4 * t * t)
                Y  = Wc + (Wc - preWc) * (numerator / t)
                preWc = Wc
                if (MatUtil.norm(grad)  < tol * normWc && (nIter > 4)) {
                    break()
                }
            }
        }
        res.Wc = Wc
    }

}
