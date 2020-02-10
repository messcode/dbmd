package com.utils

import breeze.linalg.{DenseMatrix, DenseVector, max, sum}
import breeze.stats.distributions._
import org.apache.spark.mllib.linalg.{Vector, Vectors}

import scala.util.Random


object  AddNoiseMat{
    type MatTuple = ((Int, Int), DenseMatrix[Double])
    type VectorIterator = Iterator[Vector]
    def addGaussian(mat: DenseMatrix[Double],  sd: Double = 0.01): DenseMatrix[Double] = {
        val Gaussian = breeze.stats.distributions.Gaussian(0.0, 1.0)
        val noisyMat = mat + DenseMatrix.rand[Double](mat.rows, mat.cols, Gaussian) * sd
        noisyMat
    }

    def addGaussianMixture(matTuple: MatTuple, stdVector: DenseVector[Double], mixtureProp: DenseVector[Int]): MatTuple = {
        val (indexTuple, mat) = matTuple
        assert(sum(mixtureProp) == 10)
        var noisyMat = mat
        if (indexTuple._1 <= 2) {
            noisyMat = addGaussian(mat, stdVector(0))
        } else if (indexTuple._1 <= 5) {
            noisyMat = addGaussian(mat, stdVector(1))
        } else {
            noisyMat = addGaussian(mat, stdVector(2))
        }
        (indexTuple, noisyMat)
    }
}

object AddNoiseVector {
    def addGaussian(vec: Vector, sd: Double=0.01): Vector = {
        val r = new Random()
        val noisyVec = DenseVector((1 to vec.size map(_ => r.nextGaussian() * sd)).toArray) + DenseVector(vec.toArray)
        Vectors.dense(noisyVec.toArray)
    }

    def addGaussianMixture(vec: Vector, stdVector: DenseVector[Double],
                    mixtureProp: DenseVector[Double]): Vector = {
        val r = new Random()
        val p = r.nextDouble()
        var noisyVec = vec
        if (p <= 0.3) {
            noisyVec = addGaussian(vec, stdVector(0))
        } else if (p <= 0.6) {
            noisyVec = addGaussian(vec, stdVector(1))
        } else {
            noisyVec = addGaussian(vec, stdVector(2))
        }
        noisyVec
    }
}

object NoiseSituation {
    case class Situation(stdVector: DenseVector[Double], mixtureProp: DenseVector[Int])
    case class ComSituation(stdVector: DenseVector[Double], mixtureProp: DenseVector[Double])
    def getSituation(std: Double): Situation = {
        Situation(DenseVector(1d, 1.5d, std), DenseVector(3, 3, 4))
    }

    def getComSituation(std: Double): ComSituation = {
        ComSituation(DenseVector(0.1d, 1.5d, std), DenseVector(.3, .3, .4))
    }
}

