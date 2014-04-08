package algorithm

object Kernel {

  //高斯核函数
  def GuassionKernel(xj: LabeledFeature,
    xi: LabeledFeature): Double = {
    val sum = 0.0
    val zips = xj.features.zip(xi.features)
    val va = zips.map(t => (t._1.toDouble - t._2.toDouble)
      * (t._1.toDouble - t._2.toDouble))
    math.exp(-1 * va.reduce(_ + _) / 0.4)
  }

  //线性核函数
  def linearKernel(xj: LabeledFeature,
    xi: LabeledFeature): Double = {

    xi.features.zip(xj.features).
      map(t => t._1.toDouble * t._2.toDouble).reduce(_ + _)

  }

  //多项式核函数
  def polymonialKernel(xj: LabeledFeature,
    xi: LabeledFeature): Double = {
    val p = 3
    val sum = linearKernel(xj, xi)
    math.pow(sum + 1, p)
  }

  //内积，线性核函数
  def innerProduct(a1: LabeledFeature,
    a2: LabeledFeature): Double = {
    a1.features.zip(a2.features).map(t => {
      t._1.toDouble * t._2.toDouble
    }).sum
  }

  //RBF核函数
  def rbfKernel(xj: LabeledFeature,
    xi: LabeledFeature): Double = {
    //e^-(gamma * <x-y, x-y>^2)
    val sum = 0.0
    val zips = xj.features.zip(xi.features)
    val va = zips.map(t => (t._1.toDouble - t._2.toDouble) *
      (t._1.toDouble - t._2.toDouble))

    val gama = 0.01
    math.exp(-gama * va.reduce(_ + _))

  }
}