package classifier.logistic

import classifier.ClassifierBase
import core.Instances
import scala.collection.mutable.HashSet
import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuffer
import core.Feature
import classifier.Model
import utils.Utils
import classifier.ClassifierBase

/**
 * simple class for Logistic Regression Classification
 * support multiclass by one vs one Binary Classifiers
 * Note:
 * LR.BATCH/LR.MINI_BATCH/LR.SGD converge differently ,if the accuracy is not as expected,try different parameter
 *  learningRate try: 0.01 ,0.03 ,0.05 and up
 *  lambda try : 0.01 and the same
 *
 * <a href=""> Logistic Regression Classification </a>
 * @author Icgemu
 * @version 0.0.1
 */
class LR(
  insts: Instances, //training data
  iteation: Int, //iteration 
  learningRate: Double, //learningRate
  mode: Int, //LR.BATCH/LR.MINI_BATCH/LR.SGD
  cvFold: Int, //fold for cross validation
  L2: Boolean, //L2 regularization
  lambda: Double* //parameter ¦Ë for L2 regularization
  ) extends ClassifierBase {

  private[this] def classifier(
    input: ArrayBuffer[Feature],
    pair: (String, String)): LRModel = {

    val featureLen = input(0).features.length
    //weight inculde w0
    var initWeights = Array.fill[Double](featureLen + 1)(math.random)

    val data = input.map(p => {
      var fs = p.features.map(f => f.toDouble)
      //add 1 as the first attribute as a convenience of calculating w0
      val arr = Array(1.0, fs: _*)
      new Feature(p.i, p.target, arr.map(f => f.toString()), p.weight)
    })

    assert(data(0).features.length == initWeights.length)
    initWeights = descent(data, initWeights, pair)
    val w: Array[Double] = Array[Double](initWeights: _*)

    LRModel(w, pair)
  }

  //Gradient Descent Algorithm
  private[this] def descent(data: ArrayBuffer[Feature],
    initWeights: Array[Double], pair: (String, String)): Array[Double] = {

    for (j <- 0 until iteation) {

      val fun_ = mode match {
        case LR.BATCH => batch _
        case LR.MINI_BATCH => miniBatch _
        case LR.SGD => sgd _
      }

      fun_(data, initWeights, pair)
    }
    initWeights
  }

  /**
   * Sum the Gradient of every feature
   *
   * @param data sample data
   * @param initWeights weight currently use
   * @param pair labels pair currently use
   */
  private[this] def bathGradient(data: ArrayBuffer[Feature],
    initWeights: Array[Double],
    pair: (String, String)): Array[Double] = {
    val gradient = data.map(f => { // calculate feature's current gradient
      val label = if (f.target.equals(pair._1)) 1 else 0
      val feature = f.features.map(f => f.toDouble)

      var featureGradient = new Array[Double](initWeights.length)
      for (i <- 0 until initWeights.length) {
        val r = (Utils.h(feature, initWeights) - label) * feature(i)
        featureGradient(i) = r
      }
      featureGradient
    }).reduce((a, b) => { // sum on all features
      val t = a.zip(b)
      val r = t.map(t => t._1 + t._2)
      r
    })
    gradient
  }

  /**
   * Batch Gradient Descent Algorithm
   * use all samples to update current weight in one iteration
   *
   * @param data sample data
   * @param initWeights weight currently use
   * @param pair labels pair currently use
   */
  private[this] def batch(
    data: ArrayBuffer[Feature],
    initWeights: Array[Double],
    pair: (String, String)) {

    val gradient = bathGradient(data, initWeights, pair)

    val count = data.size.toInt
    for (i <- 0 until initWeights.length) {
      initWeights(i) = updateWeight(initWeights(i), gradient(i), count.toDouble, if (i == 0) true else false)
    }

  }
  /**
   * get a fraction of samples to update current weight
   *
   * @param data sample data to generate
   * @param fraction  limit at 0.0~1.0
   */
  private[this] def sample(data: ArrayBuffer[Feature], fraction: Double): ArrayBuffer[Feature] = {
    val size = (data.size * fraction).toInt
    var arr = new ArrayBuffer[Feature]

    for (i <- 0 until size) {
      val j = data.size * math.random.toInt
      arr += data(j)
    }
    arr
  }

  /**
   * Mini Batch Gradient Descent Algorithm
   * use a fraction of samples to update current weight in one iteration
   *
   * @param data sample data
   * @param initWeights weight currently use
   * @param pair labels pair currently use
   */

  private[this] def miniBatch(
    data: ArrayBuffer[Feature],
    initWeights: Array[Double],
    pair: (String, String)) {

    //currently use fixed fraction scale 0.3
    val sampleSize = (data.size * 0.3).toInt

    var size = data.size;
    var cur = 0;
    while (cur < size) {
      val sampleData = data.slice(cur, cur + sampleSize)
      cur += sampleSize
      val sizeA = sampleData.size;
      if (sizeA > 0) {
        val gradient = bathGradient(sampleData, initWeights, pair)

        for (i <- 0 until initWeights.length) {
          initWeights(i) = updateWeight(initWeights(i), gradient(i), sizeA, if (i == 0) true else false)
        }
      }
    }

  }

  /**
   * Stochastic Gradient Descent Algorithm
   * use one sample to update current weight in one iteration
   *
   * @param data sample data
   * @param initWeights weight currently use
   * @param pair labels pair currently use
   */

  private[this] def sgd(data: ArrayBuffer[Feature],
    initWeights: Array[Double],
    pair: (String, String)) {

    data.map(f => {
      val label = if (f.target.equals(pair._1)) 1 else 0
      val feature = f.features.map(f => f.toDouble)
      for (i <- 0 until initWeights.length) {
        val r = (Utils.h(feature, initWeights) - label) * feature(i)
        initWeights(i) = updateWeight(initWeights(i), r, 1, if (i == 0) true else false)
      }
    })
  }

  /**
   * update the weight with the learningRate
   *
   * @param oldW old Weight
   * @param newDescentRate new Descent rate
   * @param C Batch size
   * @Param firstW first attribute?
   */
  private[this] def updateWeight(oldW: Double, newDescent: Double, C: Double, firstW: Boolean): Double = {
    val t = (if (!L2 || firstW) 0.0d else lambda.head)
    val r = oldW - learningRate * (1.0 / C) * (newDescent - t * oldW)
    r
  }

  override def train(): Model = {
    //number of class
    val numClass = insts.numClass
    //data for the class
    val classData = new HashMap[String, ArrayBuffer[Feature]]

    //stratify data for every class
    for (cls <- insts.classof) {
      if (!classData.contains(cls)) classData(cls) = new ArrayBuffer[Feature]
      classData(cls) = insts.data.filter(p => p.target.equalsIgnoreCase(cls))
    }

    val cls = insts.classof
    val clsPair = makePair(cls)
    val models = ArrayBuffer[LRModel]()
    var avg = 0.0
    for (pair <- clsPair) {
      val classA = classData(pair._1)
      val classB = classData(pair._2)

      val foldA = makeFold(classA, cvFold)
      val foldB = makeFold(classB, cvFold)
      var r = 0.0

      for (i <- 0 until cvFold) {
        val traindata = mergeFold(foldA, i).++(mergeFold(foldB, i))
        val testdata = foldA(i).++=(foldB(i))
        shuffle(traindata)

        val model = classifier(traindata, pair)
        var test = new Instances(insts.numIdx);
        testdata.map(test.addInstances(_))
        r += model.predict(test)
      }

      avg += (r / cvFold)

      val alldata = classA ++ classB
      shuffle(alldata)
      val fmodel = classifier(alldata, pair)
      models += fmodel
    }

    MultiLRModel(models)
  }

  
}

private[this] case class LRModel(w: Array[Double], pair: (String, String)) extends Model {
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
      val x = f.features.map(f => f.toDouble)
      val yx = Utils.h(x, w)
      val l = if (yx <= 0.5) 0 else 1
      val rlabel = if (1 == l) pair._1 else pair._2
      matrix(label)(rlabel) = matrix(label).getOrElse(rlabel, 0) + 1
    })

    val a = matrix.map(f => f._2.values.sum).sum
    val r = (matrix(pair._1).getOrElse(pair._1, 0) + matrix(pair._2).getOrElse(pair._2, 0)) * 1.0 / a
    r
  }
}

private[this] case class MultiLRModel(models: ArrayBuffer[LRModel]) extends Model {
  override def predict(test: Instances): Double = {
    var r = 0.0
    test.data.map(f => {

      //count of this sample belong to the class label
      val matrix = new HashMap[String, Int]

      for (model <- models) {
        val w = model.w
        val pair = model.pair
        val x = f.features.map(f => f.toDouble)
        val yx = Utils.h(Array(1.0, x: _*), w)
        val l = if (yx <= 0.5) 0 else 1

        //this sample's label in this model
        val rlabel = if (1 == l) pair._1 else pair._2
        //increase the count of belonging label
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
    r / test.data.size
  }
}

object LR {

  val BATCH = 1
  val MINI_BATCH = 2
  val SGD = 3

  val L2 = true

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
    val t = new LR(insts, 100, 0.05, LR.SGD, 10, LR.L2, 0.02)
    val model = t.train()
    val accu = model.predict(insts)
    println(accu);
  }
}