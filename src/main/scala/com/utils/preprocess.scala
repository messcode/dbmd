package com.utils

import breeze.linalg.DenseVector
import org.apache.spark.mllib.linalg.{Vector, Vectors}

object TextConverter {
    def text2Arr(text: String, delimiter: String, hasLabel: Boolean):
    (Vector, Int) = {
        val arr = text.split(delimiter)
        if (hasLabel) {
            val label = arr(0).toDouble
            val arr_out = Vectors.dense(arr.slice(1, arr.length).map(_.toDouble))
            (arr_out, label.toInt)
        } else {
            val label = -1
            val arr_out = Vectors.dense(arr.map(_.toDouble))
            (arr_out, label)
        }
    }

    def text2Vec(text: String, delimiter: String, hasLabel: Boolean):
    (DenseVector[Double], Int) = {
        val arr = text.split(delimiter)
        if (hasLabel) {
            val label = arr(0).toDouble
            val arr_out = DenseVector(arr.slice(1, arr.length).map(_.toDouble))
            (arr_out, label.toInt)
        } else {
            val label = -1
            val arr_out = DenseVector(arr.map(_.toDouble))
            (arr_out, label)
        }
    }
}
