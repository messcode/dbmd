package com.utils

import breeze.linalg.{*, Axis, DenseMatrix, DenseVector, max, sum}
import breeze.numerics.{log2, sqrt}
import breeze.optimize.linear.KuhnMunkres
import org.apache.spark.rdd.RDD

import scala.language.postfixOps


class ExternalMetrics(trueLabels: RDD[Int], predLabels: RDD[Int]){

  val n = trueLabels.count().toDouble
  assert(n == predLabels.count())
  val trueCounts = trueLabels.map((_, 1)).reduceByKey(_ + _).collect()
  val predCounts = predLabels.map((_, 1)).reduceByKey(_ + _).collect()
  val trueNum = trueCounts.length
  val predNum = predCounts.length
//  println("# True class = %d" format trueNum)
//  println("# Predicted class = %d" format predNum)
//  trueCounts.foreach(x => println("#TrueLabel=%d Num=%d" format(x._1, x._2)))
//  predCounts.foreach(x => println("#PredLabel=%d Num=%d" format(x._1, x._2)))
  def getJointCounts():DenseMatrix[Int] = {
    val zippedLabels = trueLabels.collect().zip(predLabels.collect())
    val jointCounts = DenseMatrix.zeros[Int](trueNum, predNum)
    for ( ((l1, _), i) <- trueCounts.zipWithIndex; ((l2, _), j) <- predCounts.zipWithIndex) {
      jointCounts(i, j) = zippedLabels.filter{case(l, p) => l==l1 && p==l2}.length
    }
    jointCounts
  }
  val jointCounts = getJointCounts()
//  println(jointCounts.toString())
  if (jointCounts.rows != jointCounts.cols) {
    println("WARNING: Class numbers (%d) of prediction and ground truth (%d) don't match! ".
      format(jointCounts.cols, jointCounts.rows))
  }

  val weightSeq = jointCounts(*, ::).map(x => x.map(transformCount(_)).toArray.toSeq).toArray.toSeq
  val ret = KuhnMunkres.extractMatching(weightSeq)

  def transformCount(c:Int): Double = {
    if (c == 0){
      1d
    } else {
      val freq = c.toDouble / n
      1 / (1 + freq)
    }
  }

  def accuracy(): Double = {
    // prepare cost matrix
    val acc = (ret._1).zipWithIndex.map( x => if (x._1 < 0) 0d else jointCounts(x._2, x._1)).sum / n
    acc
  }

  def fMeasure(): Double = {
    val predLabels = sum(jointCounts(*, ::))
    val trueLabels = sum(jointCounts(::, *))
    val precision = (ret._1).zipWithIndex.map(x =>  {
      if (x._1 < 0) 0d
      else (jointCounts(x._2, x._1).toDouble / predLabels(x._2).toDouble)
    })//.reduce(_ + _) / predLabels.size.toDouble
    val recall = (ret._1).zipWithIndex.map(x => {
      if (x._1 < 0) 0d
      else (jointCounts(x._2, x._1).toDouble / trueLabels(x._1).toDouble)
    })//.reduce(_ + _) / trueLabels.inner.size.toDouble
    val f1 = (precision zip recall).collect{case x if (x._1 + x._2 > 0)  => 2 * x._1 * x._2 / (x._1 + x._2)
    }
    //val f1 = 2d * (precision * recall) / (precision + recall)
    println("sum(f1) / f1.size=%f" format f1.reduce(_ + _) / f1.size.toDouble)
    val p = precision.reduce(_ + _) / precision.size.toDouble
    val r = recall.reduce(_ + _) / recall.size.toDouble
    println("average and compute f1=%f" format 2 * (p * r) / (p + r))
    f1.reduce(_ + _) / f1.size.toDouble
  }
}

