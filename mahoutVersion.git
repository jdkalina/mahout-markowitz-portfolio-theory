%spark 

val df = spark.read.format(\"csv\").option(\"header\", \"true\").load(\"data.csv\")
val df2 = df.drop(\"Trade Date\")\ndf2.show()


import org.apache.mahout.math._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.drm._
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.drm.RLikeDrmOps._
import org.apache.mahout.sparkbindings._
implicit val sdc: org.apache.mahout.sparkbindings.SparkDistributedContext = sc2sdc(sc)

import org.apache.spark.mllib.linalg.{Vectors => SparkVector}
val returnsDRM = drmWrapDataFrame(df = df2)


import collection._
import JavaConversions._

// ^^ for matrix iteration

val number_of_portfolios = 1000
val number_of_assets_in_each_portfolio = df2.columns.size

val inCoreWeights = Matrices.uniformView(number_of_portfolios, number_of_assets_in_each_portfolio, 1)
val normalizedInCoreWgts = dense((for (row <- inCoreWeights) yield row / row.sum).toArray)
val weightsDRM = drmParallelize(normalizedInCoreWgts)

ExpectedVolatility = \\\\(W^{T} \\cdot (\\Sigma \\cdot W) \\\\)

\\\\(\\Sigma\\\\) - is the covariance matrix for the (log) returns on the assets in the portfolio
\\\\(W\\\\) - is a matrix of portfolios where each row is a portfolio represented by a number of weights of each asset (columns)

val thing = dcolMeanCov(returnsDRM)
val returnMeans = thing._1\nval sigmaDRM = thing._2 * 252


val portfolioCoVarianceDRM = (weightsDRM %*% ( sigmaDRM %*% weightsDRM.t )   )
// get diag vector


val pcvIC = portfolioCoVarianceDRM.collect
val evVec = (dvec((0 until pcvIC.nrow).map(i => Math.sqrt(pcvIC(i,i)) )))
// stdv of ^^\nprint(\"\\n\\n\" + evVec.length)
// ta-da we hav ewhat we want

// ret_arr[ind] = np.sum( (na_eq_log.mean() * weights) * 252)
val retVec = (weightsDRM %*% returnMeans * 252).collect(::,0) // todo better name


print(\"\\n\\n\" + retVec.length)
  
// #Sharpe Ratio
// sharpe_arr[ind] = ret_arr[ind]/vol_arr[ind] 
val sharpRatVec = retVec / evVec

import org.apache.mahout.math.scalabindings.MahoutCollections._
val bestPortfolioIndex = sharpRatVec.toArray.zipWithIndex.sortBy(_._1).takeRight(1).map(_._2).toList(0)


normalizedInCoreWgts.toArray.toList(bestPortfolioIndex)

// match these to stocks. 
