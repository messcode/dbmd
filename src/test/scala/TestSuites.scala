import breeze.linalg._
import breeze.numerics.abs
import breeze.stats.distributions.Dirichlet
import com._
import com.dbmd.Mappers
import com.utils.MatUtil
import org.scalatest._

class TestMatUtil extends FlatSpec {
   "MatUtil.softh" should "shrinkage the entries in matrix" in {
       val m1 = DenseMatrix.zeros[Double](2, 2)
       m1(0, 0) = -1.5d
       val m2 = MatUtil.softh(m1, .1)
       assert(m1(0, 0) + 0.1d == m2(0, 0))
   }

}

class TestMappers extends FlatSpec {
    /**
     * (index, Xc, Wc, Hc, Uc, vars, \delta Hc)
     */
    val m = 5
    val rank = 2
    val n = 100
    val alpha = DenseVector.ones[Double](rank)
    val Wc =  DenseMatrix.rand[Double](m, rank)
    val W0 = DenseMatrix.rand[Double](m, rank)
    val dirchilet = Dirichlet(alpha.toArray).sample(n)
    val Hc: DenseMatrix[Double] = DenseMatrix.zeros[Double](rank, n)
    for (i <- dirchilet.indices) {
        Hc(::, i) := dirchilet(i)
    }
    val Xc = Wc * Hc
    val Z = DenseMatrix.zeros[Double](m, rank)
    val l = ((1, 1), Xc, W0, Hc, Hc * Hc.t, 1d, 0d)

    def calObjval(l:idRes, lam:Double): Double = {
        val objVal = .5 * MatUtil.squareNorm(l._2 - l._3 * l._4) + sum(abs(l._2)) * lam
        objVal
    }

    "Mappers.fista" should "reduce the value of the objective function" in {
        //(index, Xc, Wc, Hc, HcHct, vars, \delta Hc)
        val lam = 0.05d
        val objVal1 = calObjval(l, lam)
        val gradF = (Z * l._5 - l._2 * l._4.t)
        val nextl = Mappers.fista(l, Z, lam, 0.01, gradF, tol=1e-4)
        val objVal2 = calObjval(nextl, lam)
        println("Objval1=%.4f Objval2=%.4f" format(objVal1, objVal2))
        val minl = (l._1, l._2, Wc, l._4, l._5, l._6, l._7)
        println("Min objval = %.4f" format(calObjval(minl, lam)))
        assert(objVal2 < objVal1)
    }
}
