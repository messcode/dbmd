import breeze.linalg.{DenseMatrix, DenseVector}
import com._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

package object com {
    type Mat = DenseMatrix[Double]
    type BcastMat = Broadcast[Mat]
    type Vec = DenseVector[Double]
    /**
     * (index, Xc, Wc, Hc, Uc, vars, \delta Hc)
     */
    type idRes = ((Int, Int), Mat, Mat,
        Mat, Mat, Double, Double)
    type idResRDD = RDD[idRes]
    type indexedMat = RDD[((Int, Int), Mat)]
    type hyperParameter = (String, Double)
}
