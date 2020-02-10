package com.dbmd
case class DBMDConfig(
                   input: String = null,
                   node: Int = 1,
                   output: String = null,
                   rank: Int = 1,
                   lam: Double = 0.5,
                   hp: Double = 5.0,
                   maxIter: Int = 20,
                   hasLabel: Boolean = false,
                   delimiter: String = ",",
                   tol: Double = 1e-3,
                   cv: Boolean = false,
                   alpha: Double = 0.1d,
                   seed: Long = 0,
                   init_method: String = "randomCols",
                   std: Double = -1,
                   algorithm: String = "ADM"
                 )