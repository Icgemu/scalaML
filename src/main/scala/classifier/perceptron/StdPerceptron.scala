package classifier.perceptron

import classifier.ClassifierBase
import core.Instances
import classifier.Model
import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuffer
import core.Feature
import scala.collection.mutable.HashSet

class StdPerceptron(
  insts: Instances,
  learningRate: Double,
  iteration: Int,
  cvFold: Int) extends ClassifierBase {

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

    val models = ArrayBuffer[InterModel]()
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
        val w = Array.fill(insts.attr)(0.0)
        val bias = 0.0
        val model = train0(traindata, w, bias, pair)
        var test = new Instances(insts.numIdx);
        testdata.map(test.addInstances(_))
        r += model.predict(test)
      }
      avg += (r / cvFold)
      val w = Array.fill(insts.attr)(0.0)
      val bias = 0.0
      val alldata = classA ++ classB
      shuffle(alldata)
      val fmodel = train0(alldata, w, bias, pair)
      //val fmodel = trainByGram(alldata, pair, rate, T)
      models += fmodel
    }
    PerceptronModel(models)
  }

  def train0(data: ArrayBuffer[Feature],
    w: Array[Double],
    bias: Double,
    pair: (String, String)): InterModel = {
    var b = bias
    var err = data.size
    var j = iteration
    while (err > 0 && j > 0) {
      //println(err)
      err = 0
      data.map(f => {
        val yi = if (f.target.equalsIgnoreCase(pair._1)) 1 else -1
        val yx = yi * (w.zip(f.features).map(t => t._1 * t._2.toDouble).sum + b)
        if (yx <= 0) {
          for (i <- 0 until w.size) {
            w(i) = w(i) + learningRate * yi * f.features(i).toDouble
          }
          b = b + learningRate * yi
          err += 1
        }
      })
      j -= 1
    }

    //Model(b, w, pair, null)
    InterModel(Array(b, w: _*), pair)
  }

}

case class InterModel(w: Array[Double], pair: (String, String)) extends Model {

  override def predict(test: Instances): Double = {
    //rewrite the test data add 1 to the first attribute
    var data = test.data.map(p => {
      var fs = p.features.map(f => f.toDouble)
      val arr = Array(1.0, fs: _*)
      new Feature(p.i, p.target, arr.map(f => f.toString()), p.weight)
    })
    //val data = test.data

    val matrix = new HashMap[String, HashMap[String, Int]]
    matrix(pair._1) = new HashMap[String, Int]
    matrix(pair._2) = new HashMap[String, Int]

    data.map(f => {
      val label = f.target
      val yx = (w.zip(f.features).map(t => t._1 * t._2.toDouble).sum)
      val l = if (yx <= 0) -1 else 1
      val rlabel = if (1 == l) pair._1 else pair._2
      matrix(label)(rlabel) = matrix(label).getOrElse(rlabel, 0) + 1
    })

    val a = matrix.map(f => f._2.values.sum).sum
    val r = (matrix(pair._1).getOrElse(pair._1, 0) + matrix(pair._2).getOrElse(pair._2, 0)) * 1.0 / a
    return r
  }
}

case class PerceptronModel(models: ArrayBuffer[InterModel]) extends Model {

  override def predict(test: Instances): Double = {
    //rewrite the test data add 1 to the first attribute
    var data = test.data.map(p => {
      var fs = p.features.map(f => f.toDouble)
      val arr = Array(1.0, fs: _*)
      new Feature(p.i, p.target, arr.map(f => f.toString()), p.weight)
    })
    var r = 0.0
    data.map(f => {

      //count of this sample belong to the class label
      val matrix = new HashMap[String, Int]

      for (model <- models) {
        //val b = model.b
        val w = model.w
        val pair = model.pair

        val yx = (w.zip(f.features).map(t => t._1 * t._2.toDouble).sum)
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

object StdPerceptron {

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
    val t = new StdPerceptron(insts,0.05,100,10)
    val model = t.train()
    val accu = model.predict(insts)
    println(accu);
  }
}