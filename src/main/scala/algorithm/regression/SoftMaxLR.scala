
package algorithm.regression

import algorithm.Instances
import scala.collection.mutable.HashMap
import algorithm.LabeledFeature
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashSet

//http://www.cnblogs.com/frog-ww/archive/2013/01/06/2846988.html
object SoftMaxLR {

  case class Model(theta: Array[Array[Double]])

  var K = 0 //类数量 -1
  var M = 0 //实例数
  var N = 0 //特征个数，第一个特征值为1

  val labelsTOInt = new HashMap[String, Int]
  val intTlabels = new HashMap[Int, String]
  def classifier(insts: Instances, learnRate: Double, momentum: Double, ite: Int): Model = {

    K = insts.classof.size - 1
    M = insts.data.size
    N = insts.attr + 1

    val data = insts.data.map(lf => {
      val label = lf.label
      val index = insts.classToInt(lf)
      labelsTOInt(label) = index
      intTlabels(index) = label
      val ff = lf.features.map(f => f.toDouble)
      val arr = Array(1.0, ff: _*)
      new LabeledFeature(lf.i, label, arr.map(f => f + ""), lf.weight)
    })

    val h_value = Array.fill(K)(0.0)
    var theta = Array.fill(h_value.length)(Array.fill(N)(math.random))

    for (i <- 0 until ite) {
      val alpha = (learnRate) / (i + learnRate)
      modify_stochostic(data, learnRate, momentum, h_value, theta)
      //modify_batch(data, learnRate, momentum, h_value, theta)
    }
    Model(theta)
  }

  def fun_eqx(x: LabeledFeature, q: Array[Double]): Double = {

    val r = x.features.zip(q).map(t => t._1.toDouble * t._2).sum

    val d = math.pow(math.E, r)
    //println(d)
    d
  }

  def h(x: LabeledFeature, h_value: Array[Double],
    theta: Array[Array[Double]]) = {
    var sum = 1.0

    for (i <- (0 until K)) {
      h_value(i) = fun_eqx(x, theta(i))
      sum += h_value(i)
    }

    //assert(sum != 0)

    for (i <- (0 until K)) {
      h_value(i) = h_value(i) / sum

    }
  }

  def modify_stochostic(x: ArrayBuffer[LabeledFeature], learnRate: Double, momentum: Double, h_value: Array[Double],
    theta: Array[Array[Double]]) {
    for (j <- (0 until M)) {

      h(x(j), h_value, theta)

      for (i <- (0 until K)) {

        for (k <- (0 until N)) {
          val yi = (if (labelsTOInt(x(j).label) == i) 1 else 0)
          val xi = (x(j).features(k).toDouble)
          val b = theta(i)(k)
          theta(i)(k) += learnRate * (xi * (yi - h_value(i)) + momentum * theta(i)(k))
          //            if(theta(i)(k) < Double.MaxValue){
          //            println(b-theta(i)(k))
          //           }
        }
      }
    }
  }

  def modify_batch(x: ArrayBuffer[LabeledFeature], learnRate: Double, momentum: Double, h_value: Array[Double],
    theta: Array[Array[Double]]) {
    for (i <- (0 until K)) {
      var sum = new Array[Double](N)
      for (j <- 0 until M) {
        h(x(j), h_value, theta)
        for (k <- 0 until N) {
          sum(k) += (x(j).features(k).toDouble) *
            ((if (labelsTOInt(x(j).label) == i) 1 else 0) - h_value(i))
        }
      }
      for (k <- 0 until N) {
        //theta(i)(k) += 0.001 * sum(k) / N
        theta(i)(k) += learnRate * (sum(k) + momentum * theta(i)(k)) / M
        println(theta(i)(k))
      }
    }

  }

  def predict(model: Model, lf: LabeledFeature) {
    //输出预测向量
    //    for (i <- 0 until K) {
    //      h_value(i) = 0
    //    }

    //train
    val h_value = Array.fill(K)(0.0)
    h(lf, h_value, model.theta)

    System.out.println(labelsTOInt(lf.label) + "=>" +
      h_value(0) + " " + h_value(1) + " " + (1 - h_value(1) - h_value(0)))
  }

  def main(args: Array[String]): Unit = {

    var numIdx = new HashSet[Int]
    numIdx.+=(0)
    numIdx.+=(1)
    numIdx.+=(2)
    numIdx.+=(3)
    //    numIdx.+=(5)
    //    numIdx.+=(7)
    //    numIdx.+=(8)
    //    numIdx.+=(10)
    val insts = new Instances(numIdx)
    insts.read("E:/books/spark/ml/decisionTree/iris.csv")

    var x = Array[Array[Double]](
      Array(47, 76, 24),
      Array(46, 77, 23),
      Array(48, 74, 22),
      Array(34, 76, 21),
      Array(35, 75, 24),
      Array(34, 77, 25),
      Array(55, 76, 21),
      Array(56, 74, 22),
      Array(55, 72, 22))
    val y = Array(1, 1, 1, 2, 2, 2, 3, 3, 3)
    //insts.read(x,y)
    val model = classifier(insts, 0.01, 0.01, 100)
    val data = insts.data.map(lf => {
      val label = lf.label
      //val index = insts.classToInt(lf)
      //labelsTOInt(label) = index
      //intTlabels(index) = label
      val ff = lf.features.map(f => f.toDouble)
      val arr = Array(1.0, ff: _*)
      new LabeledFeature(lf.i, label, arr.map(f => f + ""), lf.weight)
    })
    data.map(f => {
      predict(model, f)
    })

  }

}