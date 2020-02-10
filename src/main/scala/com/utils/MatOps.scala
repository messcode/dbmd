package com.utils

import java.io.File

import breeze.linalg.{DenseMatrix, DenseVector, csvwrite, eigSym, sum}
import breeze.numerics.{abs, signum, sqrt}
import com._

object MatUtil{
  /**
    *
    * Matrix Utilization tools.
    */
  def norm(m: DenseMatrix[Double]): Double = {
    sqrt(sum(m *:* m))
  }

  def normAvg(m: DenseMatrix[Double]): Double = {
    norm(m) / sqrt(m.size)
  }

  def squareNorm(m: DenseMatrix[Double]): Double = {
    sum(m *:* m)
  }

  def sparseness(m: DenseMatrix[Double]): Double = {
    val sqrt_n = sqrt(m.size.toDouble)
    (sqrt_n - sum(abs(m)) / norm(m)) / (sqrt_n - 1d)
  }

  def softh(m: DenseMatrix[Double], lamba: Double): DenseMatrix[Double] = {
    signum(m) *:* (abs(m) - lamba).mapValues(v => if (v > 0d) v else 0d)
  }

  def csvWriter(m: DenseMatrix[Double], fileName: String) : Unit = {
      csvwrite(new File(fileName), m, separator = ' ')
  }

  def maxSymEigval(m: DenseMatrix[Double]): Double = {
      val Lp = eigSym(m).eigenvalues(-1)
      Lp
  }

}
