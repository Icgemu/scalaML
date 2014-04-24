package classifier.perceptron

import core.Instances
import core.Feature
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashMap
import classifier.Model
import scala.collection.mutable.HashSet
import core.Kernel._

//http://alex.smola.org/teaching/pune2007/pune_3.pdf
class KernelPerceptron(
  insts: Instances,
  learningRate: Double,
  iteration: Int,
  cvFold: Int,
  kernel: (Feature, Feature) => Double) extends StdPerceptron(insts, learningRate, iteration, cvFold) {

  val gram = insts.data.map(lf1 => {
    insts.data.map(lf2 => {
      kernel(lf1, lf2)
    })
  })

  override def train(): Model = {

    val numClass = insts.numClass
    val classData = new HashMap[String, ArrayBuffer[Feature]]

    for (cls <- insts.classof) {
      if (!classData.contains(cls)) classData(cls) = new ArrayBuffer[Feature]
      classData(cls) = insts.data.filter(p => p.target.equalsIgnoreCase(cls))
    }

    val cls = insts.classof
    val matrixs = new HashMap[String, HashMap[String, Int]]
    val clsPair = makePair(cls)

    for (pair <- clsPair) {
      val p = if (pair._1 < pair._2) {
        pair
      } else {
        pair.swap
      }
    }

    val models = ArrayBuffer[KernelInterModel]()
    var avg = 0.0
    for (pair <- clsPair) {
      //println(pair)
      val classA = classData(pair._1)
      val classB = classData(pair._2)

      val foldA = makeFold(classA, cvFold)
      val foldB = makeFold(classB, cvFold)

      var r = 0.0
      for (i <- 0 until cvFold) {
        val traindata = mergeFold(foldA, i) ++ mergeFold(foldB, i)
        val testdata = foldA(i) ++ foldB(i)
        shuffle(traindata)

        val model = train0(traindata, pair)
        var test = new Instances(insts.numIdx);
        testdata.map(test.addInstances(_))
        r += model.predict(test)
      }
      avg += (r / cvFold)

      val alldata = classA ++ classB
      shuffle(alldata)
      val fmodel = train0(alldata, pair)
      //val fmodel = trainByGram(alldata, pair, rate, T)
      models += fmodel
    }
    KernelPerceptronModel(models)
  }
  def train0(data: ArrayBuffer[Feature],
    pair: (String, String)): KernelInterModel = {
    var b = 0.0
    var err = data.size
    var j = iteration
    val alpha = Array.fill(data.size)(0.0)
    val y: Array[(Int, Int)] = Array.fill(data.size)(null)
    while (err > 0 /**&& j>0*/ ) {
      //println(err)
      err = 0
      //data.map(f => {
      for (m <- 0 until data.size) {
        val f = data(m)
        val i = f.i
        val yi = if (f.target.equalsIgnoreCase(pair._1)) 1 else -1
        var sum = 0.0
        for (n <- 0 until data.size) {
          val f2 = data(n)
          val j = f2.i
          val yj = if (f2.target.equalsIgnoreCase(pair._1)) 1 else -1
          sum += alpha(n) * yj * gram(i)(j)
        }
        val yx = yi * (sum + b)
        y(m) = (yi, i)
        if (yx <= 0) {
          alpha(m) = alpha(m) + learningRate
          b = b + learningRate * yi
          err += 1
        }
      }
      //})
      j -= 1
    }

    KernelInterModel(b, alpha, pair, y)
    //Model(bias,w,pair,null)
  }

  case class KernelInterModel(
      b: Double, 
      alpha: Array[Double], 
      pair: (String, String), 
      y: Array[(Int, Int)]) extends Model {

    override def predict(test: Instances): Double = {

      val data = test.data

      val matrix = new HashMap[String, HashMap[String, Int]]
      matrix(pair._1) = new HashMap[String, Int]
      matrix(pair._2) = new HashMap[String, Int]
      for (m <- 0 until data.size) {

        val f = data(m)
        val i = f.i
        val label = f.target

        var sum = 0.0
        for (n <- 0 until alpha.size) {
          sum += alpha(n) * y(n)._1 * gram(i)(y(n)._2)
        }
        val yx = (sum + b)
        val l = if (yx <= 0) -1 else 1
        val rlabel = if (1 == l) pair._1 else pair._2
        matrix(label)(rlabel) = matrix(label).getOrElse(rlabel, 0) + 1
        //})
      }

      val a = matrix.map(f => f._2.values.sum).sum
      val r = (matrix(pair._1).getOrElse(pair._1, 0) + matrix(pair._2).getOrElse(pair._2, 0)) * 1.0 / a
      return r
    }
  }
  case class KernelPerceptronModel(models: ArrayBuffer[KernelInterModel]) extends Model {

    override def predict(test: Instances): Double = {

      val data = test.data
      var r = 0.0
      data.map(f => {

        //count of this sample belong to the class label
        val matrix = new HashMap[String, Int]

        for (model <- models) {
          val b = model.b
          val alpha = model.alpha
          val pair = model.pair
          val y = model.y

          var sum = 0.0
          for (n <- 0 until alpha.size) {
            sum += alpha(n) * y(n)._1 * gram(f.i)(y(n)._2)
          }
          val yx = (sum + b)
          // val yx =  (w.zip(f.features).map(t => t._1 * t._2.toDouble).sum + b)
          val l = if (yx <= 0) -1 else 1
          val rlabel = if (1 == l) pair._1 else pair._2
          matrix(rlabel) = matrix.getOrElse(rlabel, 0) + 1
        }

        //sort the label's count by Descendant order 
        val arr = matrix.toArray.sortBy(f => { f._2 }).reverse

        println(f.target + "=>" + matrix)
        //chose the first label as this sample's prediction
        val label = arr(0)._1

        if (f.target.equalsIgnoreCase(label)) r += 1.0
      })

      //accuracy of the prediction
      r / data.size
    }
  }
}
object KernelPerceptron {

  val GUASS = GuassionKernel _
  val LINEAR = linearKernel _
  val POLYMONIAL = polymonialKernel _
  val RBF = rbfKernel _

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
    val t = new KernelPerceptron(insts, 0.05, 100, 10, GUASS)
    val model = t.train()
    val accu = model.predict(insts)
    println(accu);
  }
}