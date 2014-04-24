package classifier.svm

import classifier.Classifier
import classifier.Model
import core.Instances
import scala.collection.mutable.ArrayBuffer
import core.Feature
import scala.collection.mutable.HashSet
import core.Kernel._

/**
 * simple class for SVM SMO algorithm Classification
 * support multiclass by one vs one Binary Classifiers
 * Note:
 * chose the second alpha has some random selection ,try more times to get the best result!
 *
 * <a href="http://research.microsoft.com/en-us/um/people/jplatt/smo-book.pdf"> Sequential minimal optimization </a>
 * @author Icgemu
 * @version 0.0.1
 */
class SMO(
  insts: Instances,//training data
  C: Double,//penalty parameter for nonlinear separable solution
  tol: Double,//tolerance for zero
  kernelFun:(Feature,Feature)=>Double //kernel function
  ) extends Classifier {

  val data = insts.data.groupBy(f => f.target)

  /**
   * Precalculate the Kernel function data K(xi,xj)
   */
  def initKernel(data: ArrayBuffer[Feature]): ArrayBuffer[ArrayBuffer[Double]] = {
    data.map(lf1 => {
      data.map(lf2 => { 
        kernelFun(lf1, lf2) 
       })
    })
  }

  override def train(): Model = {
    val labels = data.keySet.toList

    val models = new ArrayBuffer[InterModel]()
    for (i <- 0 until labels.length) {
      for (j <- i + 1 until labels.length) {
        val d = data(labels(i)) ++ data(labels(j))

        //relabeled training data
        val trainData = d.map(f => {
          val l = if (f.target == labels(i))  "1" else "-1"
          new Feature(f.i,l,f.features.map(f=>f),f.weight)
        })
        //model data
        val lp = new LabelPair(labels(i), labels(j), trainData)
        //model alpha
        val alpha = Array.fill(trainData.length)(0.0)
        //model bias
        val b = train0(trainData, alpha)
        //model for this pair of labels
        models += InterModel(lp, alpha.clone, b)
      }
    }

    SMOModel(models,kernelFun)
  }

  
  def train0(
    trainData: ArrayBuffer[Feature],
    alpha: Array[Double]): Double = {
    var b = 0.0
    //sample index meet up to 0<alpha<C 
    var boundAlpha = new HashSet[Int]()
    //precalculate Kernel function
    val kernelMat = initKernel(trainData)

    /**
     * calculate En(j) = g(xj)-yj = (Sum(alpha(i)*yi*kernel(xi,xj)) + b -yj)
     * across all sample data for sample x
     */
    def En(j: Int): Double = {
      var sum = 0.0
      for (i <- 0 until trainData.length) {
        sum += alpha(i) * trainData(i).target.toDouble * kernelMat(i)(j)
      }
      sum + b - trainData(j).target.toDouble
    }

    /**
     * find max j where Max(|Ej-Ei|) in dataSet where 0<alpha<C
     */
    def findMax(i: Int, Ei: Double): Int = {

      var max = 0.0
      var maxIndex = -1
      val ite = boundAlpha.iterator
      while (ite.hasNext) {
        //for (j <- 0 until trainData.length) {
        val j = ite.next
        if (j != i) {
          val Ej = En(j);
          if (Math.abs(Ei - Ej) > max) {
            max = Math.abs(Ei - Ej);
            maxIndex = j;
          }
        }
      }
      maxIndex
    }

    //select a sample different from i
    def randomSelect(i: Int): Int = {
      var j = -1
      do {
        j = (Math.random * trainData.length).toInt
      } while (i == j);
      j
    }

    /**
     * update alpha(i) and alpha(j) to new value
     * 
     */
    def update(
        i: Int, //first alpha
        j: Int, //second alpha
        Ei: Double, //first alpha's current En(i)
        Ej: Double, //second alpha's current En(j)
        oldAi: Double,//first alpha's current value
        oldAj: Double, //second alpha's current value
        L: Double, //low bound for new value
        H: Double, //high bound for new value
        eta: Double //eta = Kii+Kjj-2Kij
        ): Int = {
      //label is 1 or-1
      val yj = trainData(j).target.toDouble
      val yi = trainData(i).target.toDouble
      
      //update aj
      alpha(j) = alpha(j) + yj * (Ei - Ej) / eta 
      
      //buffer for find max |Ei-Ej| 
      if (0 < alpha(j) && alpha(j) < C)
        boundAlpha.add(j)

      //bound the new alpha value
      if (alpha(j) < L)
        alpha(j) = L
      else if (alpha(j) > H)
        alpha(j) = H

      //update ai
      if (Math.abs(alpha(j) - oldAj) >= 1e-5) {
        alpha(i) = alpha(i) + yi * yj * (oldAj - alpha(j))
        if (0 < alpha(i) && alpha(i) < C)
          boundAlpha.add(i)

        /** b1 new £¬ b2 new */
        var b1 = b - Ei - yi * (alpha(i) - oldAi) * kernelMat(i)(i) - yj * (alpha(j) - oldAj) * kernelMat(i)(j)
        var b2 = b - Ej - yi * (alpha(i) - oldAi) * kernelMat(i)(j) - yj * (alpha(j) - oldAj) * kernelMat(j)(j)

        //update b
        if (0 < alpha(i) && alpha(i) < C)
          b = b1
        else if (0 < alpha(j) && alpha(j) < C)
          b = b2
        else
          b = (b1 + b2) / 2
        return 1
      } else {
        return 0
      }

    }
    
    val maxNoUpdate = 10
    var currentUpate = 0
    
    while (currentUpate < maxNoUpdate) {
      var alphaChangedNum = 0
      for (i <- 0 until trainData.length) {
       
        val Ei = En(i)
        val yi = trainData(i).target.toDouble
        
        //KKT violattion condition,if condition meet,some alpha new update
        if ((yi * Ei < -tol && alpha(i) < C) ||
          (yi * Ei > tol && alpha(i) > 0) ||
          (yi * Ei > (-tol) && (yi * Ei < tol) && (alpha(i) == 0.0 || alpha(i) == C))) {

          //select max |Ei- Ej|
          //Note:
          //randomSelect bring some random result, try more time to get the best!
          var j = if (boundAlpha.size > 0) findMax(i, Ei) else randomSelect(i)

          j = if (j < 0) randomSelect(i) else j
          val Ej = En(j)

          val oldAi = alpha(i)
          val oldAj = alpha(j)

          val yj = trainData(j).target.toDouble

          //0< aj < C
          //ai*yi + aj*yj = E  //(Constant)
          var L = if (yi != yj) Math.max(0, alpha(j) - alpha(i)) else Math.max(0, alpha(i) + alpha(j) - C)
          val H = if (yi != yj) Math.min(C, C - alpha(i) + alpha(j)) else Math.min(C, alpha(j) + alpha(i))

          //eta = K11+k22-K12 = ||¦µ(x1)-¦µ(x2)||^2
          val eta = kernelMat(i)(i) + kernelMat(j)(j) -2 * kernelMat(i)(j)  

          val r = if (eta > 0) {//eta must >0
            update(i, j, Ei, Ej, oldAi, oldAj, L, H, eta)
          } else {
            System.out.println("eta out i= " + i)
            0
          }
          alphaChangedNum += r
        }
      }
      if (alphaChangedNum == 0) {//see if no update anymore
        currentUpate += 1
      } else {
        currentUpate = 0
      }
      //System.out.println("iteration out")
    }
    return b
  }
}
case class LabelPair(l1: String, l2: String, data: ArrayBuffer[Feature])
case class InterModel(forlabel: LabelPair, alpha: Array[Double], b: Double)

case class SMOModel(models: ArrayBuffer[InterModel],kernelFun:(Feature,Feature)=>Double) extends Model {

  override def predict(test: Instances): Double = {

    var probability = 0.0
    var correctCnt = 0
    val total = test.data.size

    for (i <- 0 until total) {
      val xi = test.data(i)

      val result = models.map(model => {
        //val model = models(m)
        val m_a = model.alpha
        val m_b = model.b
        //val m_k = model.k

        val m_x = model.forlabel.data
        val len = m_x.size;
        var sum = 0.0;
        for (j <- 0 until len) {
          sum += m_x(j).target.toDouble * m_a(j) * kernelFun(m_x(j), xi);
        }
        sum += m_b
        val label = if (sum > 0) model.forlabel.l1 else model.forlabel.l2
        (label, 1)
      })
      val s = result.groupBy(t => t._1).map(f => { (f._1, f._2.size) })
      print(xi.target + " => ")
      var max = -1
      var index = "1"
      s.map(f => {
        //print(f._1+":"+f._2+" => ")
        if (f._2 > max) { max = f._2; index = f._1 }
      })
      print(index)

      if (index.equalsIgnoreCase(xi.target)) { correctCnt += 1 }
      println()
    }
    correctCnt * 1.0 / total
  }
}

object SMO{
  
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
    val t = new SMO(insts, 1, 0.001,SMO.GUASS)
    val model = t.train()
    val accu = model.predict(insts)
    println(accu);
  }
}