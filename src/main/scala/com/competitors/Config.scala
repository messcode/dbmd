package com.competitors

case class CompareConfig(
                          input: String = null,
                          numClass: Int = 2,
                          hasLabel: Boolean = false,
                          delimiter: String = ",",
                          node: Int = 1,
                          runKMeans: Boolean = false,
                          runNMF: Boolean = false,
                          seed: Long = 0,
                          std: Double = -1
                        )