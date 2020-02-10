//package com.dbmd
//
//import java.util.Date
//
//import breeze.linalg.{Axis, DenseMatrix, DenseVector, argmax, inv, max, min, norm, sum}
//import breeze.numerics.{abs, sqrt}
//import breeze.stats.distributions._
//import com.utils.{AddNoiseMat, ExternalMetrics, MatUtil, NoiseSituation}
//import org.apache.spark.SparkContext
//import org.apache.spark.broadcast.Broadcast
//import org.apache.spark.mllib.clustering.KMeans
//import org.apache.spark.rdd.RDD
//import org.apache.spark.mllib.linalg.Vectors
//import org.apache.spark.sql.SparkSession
//import scopt.OParser
//
//import scala.util.control.Breaks.{break, breakable}
//
//
//
//class SparkDBMD(sc: SparkContext, C:Int,
//                rank: Int, alpha: DenseVector[Double], lam: Double,
//                rho: Double)
//{/** ResTuple hold the results for each round of computation.
//  * It consists of
//  * (index, xc, wc, hc, uw, vars, new_del_Hc)
//  */
//    type resRDD = RDD[((Int, Int), DenseMatrix[Double], DenseMatrix[Double],
//                        DenseMatrix[Double], DenseMatrix[Double], Double, Double)]
//    type indexedMat = RDD[((Int, Int), DenseMatrix[Double])]
//
//    var bro_rho: Broadcast[Double] = sc.broadcast(rho)
//    val bro_lam: Broadcast[Double] = sc.broadcast(lam)
//    val bro_phi: Broadcast[(Int, Int)] = sc.broadcast((1, 1))
//    val bro_C: Broadcast[Int] = sc.broadcast(C)
//    val bro_rank: Broadcast[Int] = sc.broadcast(rank)
//    val bro_alpha: Broadcast[DenseVector[Double]] = sc.broadcast(alpha)
//
//    def getMaxDelHc(L: resRDD): Double = L.map(_._7).max()
//
//    def initVariable(sc: SparkContext, X: indexedMat, init_W: DenseMatrix[Double]): (resRDD, DenseMatrix[Double], Broadcast[DenseMatrix[Double]]) =
//    {
//        val bro_W = sc.broadcast(init_W)
//        val bro_rank: Broadcast[Int] = sc.broadcast(rank)
//        val bro_alpha: Broadcast[DenseVector[Double]] = sc.broadcast(alpha)
//        val L = X.map(x => {
//            val rank = bro_rank.value
//            val alpha = bro_alpha.value
//            val index = x._1
//            val Xc = x._2
//            val Wc: DenseMatrix[Double] = bro_W.value
//            val m = Xc.rows
//            val n = Xc.cols
//            // H从dirchilet分布中采样
//            val dirchilet = Dirichlet(alpha.toArray).sample(n)
//            val Hc: DenseMatrix[Double] = DenseMatrix.zeros[Double](rank, n)
//            for (i <- dirchilet.indices) {
//                Hc(::, i) := dirchilet(i)
//            }
//            val Uc = DenseMatrix.zeros[Double](init_W.rows, init_W.cols)
//            val variance = 1d
//            val del_Hc = 0d
//            (index, Xc, Wc, Hc, Uc, variance, del_Hc)
//        }).cache()
//        L.count()
//        X.unpersist()
//        (L, init_W, bro_W)
//    }
//
//    def updateW(L: resRDD, W: DenseMatrix[Double], broW: Broadcast[DenseMatrix[Double]], broLam: Broadcast[Double], broRho: Broadcast[Double]):
//    (resRDD, DenseMatrix[Double], Broadcast[DenseMatrix[Double]], String) = {
////        val e_abs = 10e-4
////        val e_rel = 10e-4
////        val sqrt_mn = L.map(l => sqrt(l._2.rows * l._2.cols)).first()
////        var pri_r = 0d  // prior r value
////        var r = L.map(l => MatUtil.norm(broW.value - l._3)).sum()
////        var del_r = r - pri_r // delta r value
////        var e = sqrt_mn * e_abs + C * e_rel * MatUtil.norm(W)
////        var n_iter = 0
////        var curL = L
////        var curW = W
////        var curBroW = broW
////        var traceStr = ""
////        do  {
////            // update Wc
////            val tempL = curL.map(l => Mappers.updateWc(l, broW.value, bro_lam.value, bro_rho.value)).cache()
////            tempL.count()
////            curL.unpersist()
////            curL = tempL
////            // update W
////            // update W with sigmas
////            val varsReciprocalSum = curL.map(l => 1d / l._6).reduce(_ + _)
////            val broVarsReciprocalSum = sc.broadcast(varsReciprocalSum)
////            // curW = curL.map(l => (l._3 + l._5) /:/ (l._6 * broVarsReciprocalSum.value)).reduce(_ + _)
////            val broC = sc.broadcast(C.toDouble)
////            curW = curL.map(l => (l._3 + l._5) /:/ broC.value ).reduce(_ + _)
////            curW = MatUtil.softh(curW, lam / (C.toDouble * rho))
////            curBroW = sc.broadcast(curW)
////            // update Uc
////            curL = curL.map(l => Mappers.updateUc(l, curBroW.value))
////            pri_r = r
////            r = curL.map(l => MatUtil.norm(curBroW.value - l._3)).sum()
////            del_r = abs(r - pri_r)
////            e = sqrt_mn * e_abs + C * e_rel * MatUtil.norm(W)
////            n_iter = n_iter + 1
////            traceStr = traceStr.concat("Avg(W)=%.4f\n" format MatUtil.normAvg(curW))
////        } while ( r > e && del_r > 10e-5 && n_iter < 20);
//        val e_abs = 10e-4
//        val e_rel = 10e-4
//        var nIter = 0
//        var curL = L.map(l => Mappers.setUc(l))
//        var curBroW = broW
//        var curW = W
//        val factorsH = L.map(l => inv(DenseMatrix.eye[Double](l._4.rows) + l._4 * l._4.t /:/ broRho.value))
//        do {
//            // update Wc
//            curL = L.zip(factorsH).map(l => Mappers.updateWcLazy(l._1, l._2, curBroW.value, broRho.value))
//            // update W
//            curW = curL.map(l => l._3 + l._5).reduce(_ + _) /:/ C.toDouble
//            curW = MatUtil.softh(curW, lam / (C.toDouble * rho))
//            curBroW = sc.broadcast(curW)
//            // Update Uc
//            curL = curL.map(l => Mappers.updateUc(l, curBroW.value))
//            nIter = nIter + 1
//        } while (nIter < 10)
//        val viol = curL.map(l => MatUtil.normAvg(curBroW.value - l._3)).mean()
//        val avgWc = curL.map(l => MatUtil.normAvg(l._3)).mean()
//        val avgUc = curL.map(l => MatUtil.normAvg(l._5)).mean()
//        val avgXc = curL.map(l => MatUtil.normAvg(l._2)).mean()
//        val r = curL.map(l => MatUtil.norm(curBroW.value - l._3)).sum()
//        val sqrt_mn = L.map(l => sqrt(l._2.rows * l._2.cols)).first()
//        val e = sqrt_mn * e_abs + C * e_rel * MatUtil.norm(W)
//        val msg = "Subiter=%d r=%.4f e=%.4f viol=%.4f\nAvg(W)=%.4f Avg(Wc)=%.4f Avg(Uc)=%.4f Avg(Xc)=%.4f" format(nIter, r, e, viol,
//            MatUtil.normAvg(curW), avgWc, avgUc, avgXc)
//        (curL, curW, curBroW, msg)
//    }
//
//    def updateHc(mL: resRDD, bro_W: Broadcast[DenseMatrix[Double]], bro_alpha: Broadcast[DenseVector[Double]]): resRDD =
//    {
//        val new_L = mL.map(l => Mappers.updateHc(l, bro_alpha.value)).cache()
//        new_L.count()
//        mL.unpersist()
//        new_L
//    }
//
//    def calAvgResidual(mL: resRDD): Double ={
//        val avg_res = mL.map(l => sum(abs(l._2 - l._3 * l._4)) / l._2.size.toDouble).mean()
//        avg_res
//    }
//
//    def calAvgSparseness(mL: resRDD): Double = {
//        val avg_sp = mL.map(l => MatUtil.sparseness(l._4)).mean()
//        avg_sp
//    }
//
//    def fit(sc:SparkContext, X: indexedMat,  init_W: DenseMatrix[Double], tol: Double, maxIter: Int, check: Int=1):
//    (RDD[DenseMatrix[Double]], Double, Array[Double], DenseMatrix[Double]) = {
//        /** L, mL holds the results for each round of computation.
//          * It consists of
//          * (index, xc, wc, hc, uw, vars, new_del_Hc)
//          */
//        var (curL, curW, curBroW) = initVariable(sc, X, init_W)
//        val del_Hcs = new Array[Double](maxIter)
//        var iterations = 0
//        val startTime = System.currentTimeMillis()
//        breakable {
//            for (n_iter <- 0 until maxIter) {
//                val sigmas = new DenseVector[Double](curL.map(_._6).collect())
//                val meanSigmas = sum(sigmas) / sigmas.length
//                curL = updateHc(curL, curBroW, bro_alpha)
//                println("Before UpdateW Avg(W)=%.4f" format MatUtil.normAvg(curW))
//                val L_W_broW = updateW(curL, curW, curBroW, bro_lam, bro_rho)
//                curL = L_W_broW._1
//                curW = L_W_broW._2
//                curBroW = L_W_broW._3
//                println("After UpdateW Avg(W)=%.4f" format MatUtil.normAvg(curW))
//                println("Max(Abs(W)) = %.4f" format max(abs(curW)))
//                del_Hcs(n_iter) = getMaxDelHc(curL)
////                mL = updateVars(mL, A, B)
//                iterations += 1
//                println(L_W_broW._4)
//
//                println("--------------------Iter=" + n_iter + " time: " + new Date() + "----------------------")
//                println("Sigma: max=%.4f mean=%.4f min=%.4f" format(max(sigmas), meanSigmas, min(sigmas)))
//                println("Delta_Hc       =  %.4f" format(del_Hcs(n_iter)))
//                println("Sparseness = %.4f" format calAvgSparseness(curL))
//                if (n_iter % check == 0) {
//                    println("Avg. Residual  = %.4f" format(calAvgResidual(curL)))
//                }
//
//                if (del_Hcs(n_iter) < tol)
//                    break()
//            }
//        }
////        val totalTime = (System.currentTimeMill  is() - startTime) / 1000f
////        println("Total iteration Time = " + totalTime)
//        println(" Total iterations = " + iterations)
////        println("Time for each iteration = %.4f" format(totalTime / iterations) )
//
//        val Hc = curL.map(_._4).cache()
//        Hc.count()
//        val sigmas = curL.map(x => sqrt(x._6)).sum() / C
//        val WcW: Array[Double] = curL.map(x => norm((x._3-curBroW.value).flatten())).collect()
//        curL.unpersist()
//        curBroW.unpersist()
//        (Hc, sigmas, WcW, curW)
//    }
//
//    def predict(sc: SparkContext, W: DenseMatrix[Double], hoX: indexedMat): resRDD = {
//        val bro_W = sc.broadcast(W)
//        val bro_rank: Broadcast[Int] = sc.broadcast(rank)
//        val bro_alpha: Broadcast[DenseVector[Double]] = sc.broadcast(alpha)
//        val L = hoX.map(x => {
//            val rank = bro_rank.value
//            val alpha = bro_alpha.value
//            val index = x._1
//            val Xc = x._2
//            val Wc: DenseMatrix[Double] = bro_W.value
//            val n = Xc.cols
//            val dirichlet = Dirichlet(alpha.toArray).sample(n)
//            val Hc: DenseMatrix[Double] = DenseMatrix.zeros[Double](rank, n)
//            for (i <- dirichlet.indices) {
//                Hc(::, i) := dirichlet(i)
//            }
//            val Uw = DenseMatrix.zeros[Double](rank, n)
//            val variance = 1d
//            val del_Hc = 0d
//            (index, Xc, Wc, Hc, Uw, variance, del_Hc)
//        }).cache()
//        L.count()
//        hoX.unpersist()
//        updateHc(L, bro_W, bro_alpha)
//    }
//
//    def partitionMat(X: indexedMat, ho:Int): (indexedMat, indexedMat) = {
//        val trainX = X.filter(_._1._1 != ho)
//        val hoX = X.filter(_._1._1 == ho)
//        (trainX, hoX)
//    }
//
//    def cv(sc:SparkContext, X: indexedMat, initW : DenseMatrix[Double], tol: Double, maxIter: Int, hoIndex: Int): Double ={
//        assert(hoIndex > -1 && hoIndex < C)
//        val (trainX, hoX) = partitionMat(X, hoIndex)
//        val fitRes = fit(sc, trainX, initW, tol, maxIter)
//        val predRes = predict(sc, fitRes._4, hoX)
//        val broW = sc.broadcast[DenseMatrix[Double]](fitRes._4)
//        val reconErr = predRes.map(l => MatUtil.normAvg(l._2 - broW.value * l._4)).mean()
//        println("Reconstruction error on holdout data = %.4f" format(reconErr))
//        reconErr
//    }
//
//    /**
//      * Return the indexes of maximum  as labels .
//      * Note that the number of class may be smaller than the given rank.
//      * */
//    def getLabels(Hc: RDD[DenseMatrix[Double]]): RDD[Int] ={
//        val predLabel = Hc.flatMap(argmax(_, Axis._0).t.toArray)
//        predLabel.map(_ + 1)
//    }
//}
//
//// singleton object. take inputs and run the test
//object SparkDBMD{
//    def process(sc: SparkContext, config: DBMDConfig): (RDD[((Int, Int), DenseMatrix[Double])], RDD[Int], DenseMatrix[Double]) = {
//        // 1. load data
//        def text2arr(text: String, delimiter: String, hasLabel: Boolean):
//            (DenseVector[Double], Int) = {
//            val arr = text.split(delimiter)
//            if (hasLabel) {
//                val label = arr(0).toDouble
//                val arr_out = DenseVector(arr.slice(1, arr.length).map(_.toDouble))
//                (arr_out, label.toInt)
//            } else {
//                val label = -1
//                val arr_out = DenseVector(arr.map(_.toDouble))
//                (arr_out, label)
//            }
//        }
//        val X = sc.textFile(config.input).repartition(config.node) // repatition RDD to C slices
//        X.cache()
//        var m = X.first().split(config.delimiter).length
//        if (config.hasLabel){
//            m = m - 1
//        }
//        // Construct initialized basis matrix
//
//        val W = DenseMatrix.zeros[Double](m, config.rank)
//        if (config.init_method == "randomCols"){
//            // Initialize W with random sample
//            val W_arr = X.takeSample(false, config.rank, seed = config.seed).map( w => {
//                val res = text2arr(w, config.delimiter, config.hasLabel)
//                res._1
//            })
//            W_arr.zipWithIndex.foreach{ case(el, i) => W(::, i):= el}
//        } else if (config.init_method == "kMeans"){
//            // Initialize W ith K-Means with random sample
//            val factor = 300
//            val sampleInstances = X.takeSample(false, min(config.rank * factor, 3000),
//                seed = config.seed).map( w => {val res = text2arr(w, config.delimiter, config.hasLabel)
//                Vectors.dense(res._1.toArray)})
//            val sampleInstancesRDD = sc.parallelize(sampleInstances).cache()
//
//            val model = new KMeans().setK(config.rank).setMaxIterations(100)
//                .setSeed(config.seed).run(sampleInstancesRDD)
//            val clusters = model.clusterCenters
//            sampleInstancesRDD.unpersist()
//            W := new DenseMatrix[Double](m, config.rank, clusters.flatMap(_.toArray))
//        } else {
//            throw new IllegalArgumentException("Illegal initialization method.")
//        }
//
//        val result = X.mapPartitionsWithIndex((index, iterX) => {
//            val listX = iterX.toList
//            val nc = listX.length
//            val labels = new Array[Int](nc)
//            val dist_X = DenseMatrix.zeros[Double](m, nc)
//            listX.zipWithIndex.foreach{ case(el, i) =>
//                val res = text2arr(el, config.delimiter, config.hasLabel)
//                dist_X(::, i) := res._1
//                labels(i) = res._2
//            }
//            val indexTuple = (index, nc)
//            Array(((indexTuple, dist_X), labels)).iterator
//        })
//        X.unpersist()
//        result.count() // take action
//        val features = result.map(_._1)
//        features.count()
//        val labels = result.flatMap(_._2)
//        labels.count()
//        result.unpersist()
////        val temp = labels.sortByKey().collect()
////        temp_arr.foreach(println)
//
//        (features, labels, W)
//    }
//
//    def main(args: Array[String]): Unit = {
//        // parse input
//        val builder = OParser.builder[DBMDConfig]
//        val parser1 = {
//            import builder._
//            OParser.sequence(
//                programName("SparkDBMD"),
//                head("SparkDBMD", "0.1"),
//                opt[String]('f', "input").required().
//                  action( (x, c) => c.copy(input = x)).
//                  text("path/to/input file"),
//                opt[Int]('n', name = "node").required().
//                  action( (x, c) => c.copy(node = x)).
//                  text("number of nodes to be used"),
//                opt[Int]('r', name = "rank").required().
//                  action( (x, c) => c.copy(rank = x)).
//                  text("rank of the decomposition"),
//                opt[Double]('l', "lam").required().
//                  action( (x, c) => c.copy(lam = x)).
//                  text("l1 penalized parameter"),
//                opt[Double]("rho").required().
//                  action( (x, c) => c.copy(rho = x)).
//                  text("ADMM parameter"),
//                opt[String]('o', "output").required().
//                  action( (x, c) => c.copy(output = x)).
//                  text("path/to/put file"),
//                opt[Int]( name = "maxIter").optional().
//                  action( (x, c) => c.copy(maxIter = x)).
//                  text("maximum iteration times"),
//                opt[Double]('t', name = "tol").optional().
//                  action( (x, c) => c.copy(tol = x)).text("tolerance"),
//                opt[Unit](name = "hasLabel").
//                  action( (_, c) => c.copy(hasLabel=true))
//                  .text("data matrix has label indicator"),
//                opt[Unit](name = "cv").
//                  action( (_, c) => c.copy(cv=true))
//                  .text("Run cross validation for parameter selection."),
//                opt[String](name = "delimiter").optional().action((x, c) => c.copy(delimiter = x)).
//                  text("Specify the delimiter of input file."),
//                opt[Double](name = "alpha").optional().action((x, c) => c.copy(alpha = x)).
//                    text("Dirichlet prior"),
//                opt[Long](name = "seed").optional().action((x, c) => c.copy(seed = x)).
//                    text("Random generator seed"),
//                opt[String](name = "init_method").optional().action( (x, c) => c.copy(init_method = x)).
//                    text("Initialization method: randomCols or kMeans"),
//                opt[Double](name = "std").optional.action((x, c) => c.copy(std = x)).
//                    text("proportion of noise")
//            )
//        }
//        OParser.parse(parser1, args, DBMDConfig()) match {
//            case Some(config) =>{
//                println("================DBMD v2=====================")
//                println(">>Input file: " + config.input)
//                println(">>seed = %d" format config.seed)
//                val spark = SparkSession.builder().getOrCreate()
//                val sc = spark.sparkContext
//                sc.setLogLevel("WARN")
//                val dataset = process(sc, config)
//                var X = dataset._1
//                if (config.std > 0) {
//                    println(">>Noise std = %.2f" format config.std)
//                    val s  = NoiseSituation.getSituation(config.std)
//                    X = X.map(x => AddNoiseMat.addGaussianMixture(x, s.prop, s.params, s.mixtureProp)).cache()
//                }
//                val initW = dataset._3
//                val alpha = DenseVector.fill(config.rank, config.alpha)
//                val model = new SparkDBMD(sc, config.node, config.rank, alpha, config.lam,
//                    config.rho)
//                if (config.cv) {
//                    model.cv(sc, X, initW, config.tol, config.maxIter, 0)
//                } else {
//                    val trueLabels = dataset._2
//                    val fitRes = model.fit(sc, X, initW, config.tol, config.maxIter)
//                    val est_Hc = fitRes._1
//                    val predLabels = model.getLabels(est_Hc)
//                    val metrics = new ExternalMetrics(trueLabels, predLabels)
//                    println("NMI = %.4f" format(metrics.normalizedMI()))
//                    println("Accuracy = %.4f" format (metrics.accuracy()))
//                    MatUtil.csvWriter(fitRes._4, config.output.concat("W.txt"))
////                    val x = sc.parallelize(Array(1, 1, 1, 1, 1, 1, 1, 0, 1, 1))
////                    val y = sc.parallelize(Array(1, 0, 0, 1, 1, 1, 1, 1, 1, 0))
////                    val m = new ExternalMetrics(x, y)
////                    print(m.normalizedMI())
//
//                }
//            }
//            case None =>
//                println("Input arguments is not valid")
//                System.exit(0)
//        }
//    }
//}