package classifier.nn

import classifier.Classifier
import core.Instances
import classifier.Model
import classifier.logistic.LR
import cluster.SimpleKMeans
import core.Feature
import scala.collection.mutable.HashSet
import classifier.perceptron.StdPerceptron

class RBFNetWork(
  insts: Instances,
  K: Int,
  iteration: Int,
  learningRate: Double,
  cvFold: Int) extends Classifier {

  var model: Model = null

  //base sample points
  val m = SimpleKMeans.kmean(insts.data, K)
  val centers = m.centers
  val k = m.K
  val p = m.par

  //std
  val std = p.map(f => {
    val index = f._1
    val data = f._2
    val std1 = data.map(lf => {
      val arr = lf.features.map(xi => { xi.toDouble })
      val s = arr.zip(centers(index)).map(t => {
        (t._1 - t._2) * (t._1 - t._2)
      }).sum
      s
    }).sum
    std1 / (data.size - 1)
  }).toArray

  def train(): Model = {

    val newData = generateHD(insts)

    val lr = new StdPerceptron(newData, learningRate,iteration, 10)
    model = lr.train()

    RBFModel()
  }
  // generate high dimension data set 
  def generateHD(set: Instances): Instances = {
    val data = set.data
    val x = data.map(f => {
      val t = Array.fill(centers.size)(0.0)
      for (i <- 0 until centers.size) {
        t(i) = guass(f, centers(i), std(i))
      }
      (t)
    }).toArray

    val y = data.map(f => f.target).toArray
    //high dimension data
    val newData = new Instances(insts.numIdx)
    newData.read(x, y)
    newData
  }
  //guass
  def guass(xi: Feature, c: Array[Double], std: Double): Double = {
    val x = xi.features.map(f => f.toDouble)
    val sum = x.zip(c).map(f => {
      (f._1 - f._2) * (f._1 - f._2)
    }).sum
    math.exp(-1 * sum / (2 * std))
  }

  //Reflected Sigmoidal
  def sigmoidal(xi: Feature, c: Array[Double], std: Double): Double = {
    val x = xi.features.map(f => f.toDouble)
    val sum = x.zip(c).map(f => {
      (f._1 - f._2) * (f._1 - f._2)
    }).sum
    1 / (1 + math.exp(sum / (std)))
  }

  //Inverse multiquadrics
  def multiquadrics(xi: Feature, c: Array[Double], std: Double): Double = {
    val x = xi.features.map(f => f.toDouble)
    val sum = x.zip(c).map(f => {
      (f._1 - f._2) * (f._1 - f._2)
    }).sum
    1 / math.sqrt(sum + std)
  }
  
  case class RBFModel() extends Model {
    def predict(test: Instances): Double = {
      val newData = generateHD(test)
      model.predict(newData)
    }
  }
}
object RBFNetWork {

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
    val t = new RBFNetWork(insts,  9, 1000, 0.01, 10)
    val model = t.train()
    val accu = model.predict(insts)
    println(accu);
  }
}