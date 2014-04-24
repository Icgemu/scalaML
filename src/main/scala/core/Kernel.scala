package core

object Kernel {

  //高斯核函数
  def GuassionKernel(xj: Feature,
    xi: Feature): Double = {
    val sum = 0.0
    val zips = xj.features.zip(xi.features)
    val va = zips.map(t => (t._1.toDouble - t._2.toDouble)
      * (t._1.toDouble - t._2.toDouble))
    math.exp(-1 * va.reduce(_ + _) / 0.4)
  }

  //线性核函数
  def linearKernel(xj: Feature,
    xi: Feature): Double = {

    xi.features.zip(xj.features).
      map(t => t._1.toDouble * t._2.toDouble).reduce(_ + _)

  }

  //多项式核函数
  def polymonialKernel(xj: Feature,
    xi: Feature): Double = {
    val p = 3
    val sum = linearKernel(xj, xi)
    math.pow(sum + 1, p)
  }

  //内积，线性核函数
  def innerProduct(a1: Feature,
    a2: Feature): Double = {
    a1.features.zip(a2.features).map(t => {
      t._1.toDouble * t._2.toDouble
    }).sum
  }

  //RBF核函数
  def rbfKernel(xj: Feature,
    xi: Feature): Double = {
    //e^-(gamma * <x-y, x-y>^2)
    val sum = 0.0
    val zips = xj.features.zip(xi.features)
    val va = zips.map(t => (t._1.toDouble - t._2.toDouble) *
      (t._1.toDouble - t._2.toDouble))

    val gama = 0.01
    math.exp(-gama * va.reduce(_ + _))

  }
}