package com.dbmd

import breeze.linalg.{DenseMatrix, DenseVector, all, diag, eig, inv, max, min, norm, sum}
import breeze.numerics.{abs, pow, signum, sqrt}
import breeze.stats.distributions.Rand
import com._

import util.control.Breaks.{break, breakable}
import com.utils.MatUtil

//noinspection DuplicatedCode
object Mappers {
    /** ResTuple hold the results for each round of computation.
      * It consists of
      * (index, xc, wc, hc, uw, vars, new_del_Hc)
      */
    //type idRes = ((Int, Int), DenseMatrix[Double], DenseMatrix[Double], DenseMatrix[Double], DenseMatrix[Double], Double, Double)
    def kkt_res(v: DenseVector[Double], rank: Int, a: DenseVector[Double], b: DenseVector[Double], Q: DenseMatrix[Double]):
    DenseVector[Double] = {
        val F_a = DenseVector.zeros[Double](2 * rank + 1)
        F_a(0 until rank) := (Q * v(0 until rank)).toDenseVector - DenseVector.ones[Double](rank) *:* v(rank) - v(rank + 1 to -1) - b
        F_a(rank) = sum(v(0 until rank)) - 1
        F_a(rank + 1 to -1) := v(0 until rank) *:* v(rank + 1 to -1) - a
        F_a
    }

    def _optimize_hj(alpha: DenseVector[Double], b: DenseVector[Double], hj: DenseVector[Double], J: DenseMatrix[Double], Q: DenseMatrix[Double]): DenseVector[Double] = {
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
                val Fa = kkt_res(vk, rank, alpha_, b, Q)
                // shrinkage aplha_
                alpha_ := max(alpha_ * eta, alpha)
                if (sum(abs(Fa)) < tol)
                    break()
                val d: DenseVector[Double] = Jk \ -Fa
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
/*
 * Set sigma = 1 when update Hc
 */
    def updateHc(l: idRes, alpha: DenseVector[Double]): idRes = {
        var (index, xc, wc, hc, uw, _, del_Hc) = l
        val rank = hc.rows
        val eps = 1e-5
        val o_hc = hc.copy
        def _J_Q() = {
            val Q = wc.t * wc
            val J = DenseMatrix.zeros[Double](2 * rank + 1, 2 * rank + 1)
            J(0 until rank, 0 until rank) := Q
            J(0 until rank, rank) := -1d
            J(0 until rank, (rank + 1) to -1) := -DenseMatrix.eye[Double](rank)
            J(rank, 0 until rank) := 1d
            (J, Q)
        }

        val (mJ, mQ) = _J_Q()
        val Bi = wc.t * xc
        for (idx <- 0 until xc.cols){
            hc(0 to -1, idx) := _optimize_hj(alpha, Bi(0 to -1, idx), hc(0 to -1, idx), mJ, mQ)
        }
        hc := max(hc, eps)
        val new_del_Hc = if (del_Hc != 0) norm((o_hc - hc).flatten()) / norm(o_hc.flatten()) else norm(o_hc.flatten())
        val next_vars = (MatUtil.squareNorm(xc - wc * hc) + 1.0) / (4.0 + xc.size)
        (index, xc, wc, hc, uw, next_vars, new_del_Hc)
    }
/*
 * set sigma = 1 when update Wc
 */
    def updateWc(l: idRes, W: DenseMatrix[Double],  rho: Double): idRes = {
        val (index, xc, _, hc, uc, vars, del_Hc) = l
        val wc = ((xc * hc.t + uc) /:/ rho + W) * inv(DenseMatrix.eye[Double](hc.rows) + hc * hc.t /:/ rho)
        (index, xc, wc, hc, uc, vars, del_Hc)
    }

    def updateUc(l: idRes, W: DenseMatrix[Double]): idRes = {
        val (index, xc, wc, hc, uc, vars, del_Hc) = l
        val nextUc = uc + wc - W
        (index, xc, wc, hc, nextUc, vars, del_Hc)
    }

    def fista(l: idRes, W: Mat, lam: Double, gamma: Double, gradF: Mat, tol: Double = 1e-6): idRes = {
        val (index, xc, _, hc, hht, vars, del_Hc) = l
        val gradf = (W * hht - xc * hc.t) - gradF
        val Lp = MatUtil.maxSymEigval(hht + DenseMatrix.eye[Double](hc.rows) *:* gamma)
        val maxIter = 20
        var preWc = W
        var Y = W
        var t = 1d
        var Wc = W
        val normWc = MatUtil.norm(W)
        var finalIter = -1
        breakable{
            for (nIter <- 0 until maxIter) {
                val grad = Y * hht  - xc * hc.t +  (Y - W) *:* gamma - gradf
                Wc = Y - grad /:/ Lp
                Wc = MatUtil.softh(Wc, lam / Lp)
                val numerator = t - 1
                t = .5 + sqrt(1 + 4 * t * t)
                Y  = Wc + (Wc - preWc) * (numerator / t)
                preWc = Wc
                if (MatUtil.norm(preWc - Wc) / normWc < tol && (nIter > 4)) {
                    finalIter = nIter
                    break()
                }
            }
        }
        (index, xc, Wc, hc, hht, vars, del_Hc)
    }
}




