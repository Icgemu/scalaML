package utils

object Utils {
  
  /**
  * hypothetic funtion used in Logistic Regression
  * @param xi the feature to calculating
  * @param w weights of the feature to use
  */
  def h(xi: Array[Double], w: Array[Double]): Double = {
    val v = -1 * xi.zip(w).map((t: (Double, Double)) => t._1 * t._2).sum
    val a = 1.0 / (1.0 + math.exp(v))
    a
  }

}