package classifier.logistic

import classifier.ClassifierBase
import classifier.Model
import core.Instances
import scala.collection.mutable.HashMap
import core.Feature
import scala.collection.mutable.HashSet
import scala.collection.mutable.ArrayBuffer
/**
 * simple class for Softmax Regression Classification
 * support multiclass Classifiers
 * Note:
 * SoftMax.BATCH/SoftMax.MINI_BATCH/SoftMax.SGD converge differently ,
 * if the accuracy is not as expected,try different parameter
 *  learningRate try: 0.01 ,0.03 ,0.05 and up
 *  lambda try : 0.01 and the same
 *  θ/weights for every labels is initial with random value,this can result in bad output,try more times to get the best.
 *
 * <a href="http://ufldl.stanford.edu/wiki/index.php/Softmax_Regression"> Softmax Regression </a>
 * @author Icgemu
 * @version 0.0.1
 */
class SoftMax(
  insts: Instances, //training data
  iteration: Int, //number of iteration
  learningRate: Double, //learning rate
  mode:Int, //
  lambda: Double //parameter λ for regularization
  ) extends ClassifierBase {

  val classNum = insts.classof.size - 1 //类数量 -1
  val instsSize = insts.data.size //实例数
  val attrNum = insts.attr + 1 //特征个数，第一个特征值为1

  val labelsTOInt = new HashMap[String, Int]
  val intTlabels = new HashMap[Int, String]

  override def train(): Model = {

    //add 1 as the first attribute
    val data = insts.data.map(lf => {
      val label = lf.target
      val index = insts.classToInt(lf)
      labelsTOInt(label) = index
      intTlabels(index) = label
      val ff = lf.features.map(f => f.toDouble)
      val arr = Array(1.0, ff: _*)
      new Feature(lf.i, label, arr.map(f => f + ""), lf.weight)
    })

    shuffle(data)
    
    val h_value = Array.fill(classNum)(0.0)
    
    // θ/weights initial with random value
    var w = Array.fill(h_value.length)(Array.fill(attrNum)(math.random))
    var weight = w.map(normalize( _ ))
    
    for (i <- 0 until iteration) {
      mode match {
        case SoftMax.BATCH =>batch(data, h_value, weight)
        case SoftMax.MINI_BATCH =>mini_batch(data, h_value, weight)
        case SoftMax.SGD => sgd(data, h_value, weight)
      }
      
    }

    SoftMaxModel(weight,classNum,labelsTOInt,h _)
  }
  
 def normalize(w:Array[Double]):Array[Double] = {
   val sum = w.sum
   w.map(_ / sum)
 }


  def batch(x: ArrayBuffer[Feature], h_value: Array[Double], theta: Array[Array[Double]]) {
    for (i <- (0 until classNum)) {
      var gradient = new Array[Double](attrNum)
      //update h and theta by all simple
      for (j <- 0 until instsSize) {
        h(x(j), h_value, theta)
        for (k <- 0 until attrNum) {
          gradient(k) += (x(j).features(k).toDouble) *
            ((if (labelsTOInt(x(j).target) == i) 1 else 0) - h_value(i))
        }
      }

      for (k <- 0 until attrNum) {
        theta(i)(k) += learningRate * (gradient(k)/instsSize  - lambda * theta(i)(k))
      }
      theta(i) = normalize(theta(i))
    }

  }

  def mini_batch(data: ArrayBuffer[Feature], h_value: Array[Double], theta: Array[Array[Double]]) {
    for (i <- (0 until classNum)) {
      //currently the fraction scale is fixed 0.3
      val batchSize = (instsSize * 0.3).toInt
      var cur = 0;
      while (cur < instsSize) {
        val sampleData = data.slice(cur, cur + batchSize)
        cur += sampleData.size;
        var gradient = new Array[Double](attrNum)
        //update h and theta by batch simple
        for (j <- 0 until sampleData.size) {
          h(sampleData(j), h_value, theta)
          for (k <- 0 until attrNum) {
            gradient(k) += (sampleData(j).features(k).toDouble) *
              ((if (labelsTOInt(sampleData(j).target) == i) 1 else 0) - h_value(i))
          }
        }

        for (k <- 0 until attrNum) {
          theta(i)(k) += learningRate * (gradient(k)/sampleData.size  - lambda * theta(i)(k))
        }
        theta(i) = normalize(theta(i))
      }

    }

  }

  def sgd(x: ArrayBuffer[Feature], h_value: Array[Double], theta: Array[Array[Double]]) {
    for (j <- (0 until instsSize)) {
      h(x(j), h_value, theta)
      //update h and theta every simple
      for (i <- (0 until classNum)) {
        for (k <- (0 until attrNum)) {
          val yi = (if (labelsTOInt(x(j).target) == i) 1 else 0)
          val xi = (x(j).features(k).toDouble)
          val b = theta(i)(k)
          theta(i)(k) += learningRate * (xi * (yi - h_value(i)) + lambda * theta(i)(k))
          theta(i) = normalize(theta(i))
        }
      }
    }
  }

  /**
   * calculate hypothetic h(x(i))for sample x(i)
   *
   * @param x        sample x(i)
   * @param h_value  possibility for different class
   * @param theta    weight for different class
   */
  def h(x: Feature, h_value: Array[Double],
    theta: Array[Array[Double]]) = {
    var sum = 1.0

    for (i <- (0 until classNum)) {
      h_value(i) = fun_eqx(x, theta(i))
      sum += h_value(i)
    }

    for (i <- (0 until classNum)) {
      h_value(i) = h_value(i) / sum
    }
  }

  /**
   * calculate e^(x(i)*w) for sample x(i) and weight w
   */
  def fun_eqx(x: Feature, w: Array[Double]): Double = {
    val r = x.features.zip(w).map(t => t._1.toDouble * t._2).sum
    math.pow(math.E, r)
  }
}

case class SoftMaxModel(w: Array[Array[Double]],
    K:Int,
    labelsTOInt: HashMap[String, Int],
    h:(Feature,Array[Double],Array[Array[Double]])=>Unit) extends Model {
  
  def predict(test: Instances): Double = {
    var r=0.0
    
    var data = test.data.map(p => {
      var fs = p.features.map(f => f.toDouble)
      val arr = Array(1.0, fs: _*)
      new Feature(p.i, p.target, arr.map(f => f.toString()), p.weight)
    })
    
    data.map(lf=>{
      val h_value = Array.fill(K+1)(0.0)
      h(lf, h_value, w)
      
      val index = labelsTOInt(lf.target)
      h_value(h_value.length -1) = 1.0 - h_value.sum
      
      val max = h_value.max
      
      val f_val =h_value(index)
      if(f_val == max){
        r +=1.0
      }
    })
    r/test.data.size
  }
}

object SoftMax{
  val BATCH = 1
  val MINI_BATCH = 2
  val SGD = 3
  
  
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
    val t = new SoftMax(insts, 1000, 0.01, SoftMax.SGD, 0.02)
    val model = t.train()
    val accuracy = model.predict(insts)
    println(accuracy);
  }
}