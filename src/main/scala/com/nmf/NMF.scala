/**
  * Compute the non-negative matrix factorization using the algorithm in
  * ``
  * Scalable methods for non-negative near separable tall-and-skinny matrices.
  * ``
  * The code is modified from https://github.com/alexgittens/nmfspark
  */
package com.nmf

import java.text.SimpleDateFormat
import java.util.Calendar

import org.apache.spark.mllib.linalg.{DenseMatrix, Matrix, Vector}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import com.nmf.mlmatrix.{NNLS, RowPartition, RowPartitionedMatrix, modifiedTSQR}
import breeze.linalg.{diag, DenseMatrix => BDM, DenseVector => BDV}
import org.apache.spark.rdd.RDD



class NMF {

    case class NMFDecomposition(colbasis: DenseMatrix, loadings: DenseMatrix) {
        def H: DenseMatrix = colbasis

        def W: DenseMatrix = loadings
    }

    def report(message: String, verbose: Boolean = true) = {
        val now = Calendar.getInstance().getTime()
        val formatter = new SimpleDateFormat("H:m:s")
        if (verbose) {
            println("STATUS REPORT (" + formatter.format(now) + "): " + message)
        }
    }

    def fromBreeze(mat: BDM[Double]): DenseMatrix = {
        new DenseMatrix(mat.rows, mat.cols, mat.data, mat.isTranspose)
    }

    def fullNMF(A: RowPartitionedMatrix, rank: Int): (Array[Int], BDM[Double], BDM[Double]) = {

        val (colnorms, rmat) = new modifiedTSQR().qrR(A)
        val (extremalcolindices, finalH) = xray.computeXray(rmat, rank)
        val W = A.mapPartitions(mat => {
            val newMat = BDM.zeros[Double](mat.rows, extremalcolindices.length)
            (0 until extremalcolindices.length).foreach {
                colidx => newMat(::, colidx) := mat(::, extremalcolindices(colidx))
            }
            newMat
        }).collect()
        (extremalcolindices, W, finalH * diag(BDV.ones[Double](rmat.cols) *:* colnorms))
    }

    /* returns argmin || R - R[::, colindices]*H ||_F s.t. H >= 0 */
    def computeH(R: BDM[Double], colindices: Array[Int]): BDM[Double] = {
        val RK = BDM.horzcat((0 until colindices.length).toArray.map(colidx => R(::, colindices(colidx)).asDenseMatrix.t): _*)

        val H = BDM.zeros[Double](RK.cols, R.cols)
        val ws = NNLS.createWorkspace(RK.cols)
        val ata = (RK.t * RK).toArray

        for (colidx <- 0 until R.cols) {
            val atb = (RK.t * R(::, colidx)).toArray
            val h = NNLS.solve(ata, atb, ws)
            H(::, colidx) := BDV(h)
        }
        H
    }

    def buildRPM(X: RDD[Vector]): RowPartitionedMatrix ={
        val numcols = X.first().size
        def buildRowBlock(iter: Iterator[Vector]) : Iterator[RowPartition] = {
            val mat = BDM.zeros[Double](iter.length, numcols)
            for(rowidx <- 0 until iter.length) {
                //val currow = iter.next.vector.toBreeze.asInstanceOf[BDV[Double]]
                val currow = BDV(iter.next.toArray).t
                mat(rowidx, ::) := currow
            }
            Array(RowPartition(mat)).toIterator
        }
        new RowPartitionedMatrix(X.mapPartitions(buildRowBlock))
    }

    def train(X: RDD[Vector], rank: Int): Unit = {
        report("Calling TSQR")
        val A = buildRPM(X)
        val (colnorms, rmat) = new modifiedTSQR().qrR(A)
        report("TSQR worked")
        report("normalizing the columns of R")
        val normalizedrmat = rmat * diag( BDV.ones[Double](rmat.cols) /:/ colnorms)
        report("done normalizing")
        report("starting xray")
        val (extremalcolindices, finalH) = xray.computeXray(normalizedrmat, rank)
        report("ending xray")
        println("finalH col = %d row = %d" format(finalH.cols, finalH.rows))
    }

}

object NMF{
    def computeR(X: RowMatrix): Matrix = {
        val decomposition = X.tallSkinnyQR()
        decomposition.R
    }
}
