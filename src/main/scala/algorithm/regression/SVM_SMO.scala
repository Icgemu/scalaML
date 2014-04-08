package algorithm.regression

import scala.io.Source
import java.io.File
import org.apache.spark.mllib.regression.LabeledPoint
import scala.collection.mutable.HashSet
import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuffer
import scala.Array.canBuildFrom

object SVM_SMO {

  var data = new HashMap[String, ArrayBuffer[LabeledPoint]]()

  case class LabelPair(l1: String, l2: String, x: Array[LabeledPoint])
  case class Model(forlabel: LabelPair, alpha: Array[Double], b: Double)

  var x = loadData()
  var boundAlpha = new HashSet[Int]()
  var alpha = new Array[Double](0)
  var k = init()
  val C = 1
  val tol = 0.00001
  var b = 0.0

  def loadCsv(): Array[LabeledPoint] = {
    val f = Source.fromFile(new File("D://tmp/iris.csv"))
    val x = f.getLines.toArray.map(line => {
      val l = line.split(":")

      val label = l(1) //if(l(1).equalsIgnoreCase("1")) 1 else -1

      val p = l(0).split(",").map(v => v.toDouble)

      if (!data.contains(label)) {
        data.put(label, new ArrayBuffer[LabeledPoint])
      }
      var buffer = data.get(label).get
      val lb = LabeledPoint(label.toDouble, p)
      buffer += lb
      lb
    })
    f.close
    x
  }
  // 0ne vs one MultiClass Classification
  def trainMulti(): Array[Model] = {
    loadCsv()
    val labels = data.keySet.toList
    val models = new ArrayBuffer[Model]()
    for (i <- 0 until labels.length) {
      for (j <- i + 1 until labels.length) {
        val xi = data.get(labels(i)).get
        val xj = xi ++= data.get(labels(j)).get

        val p = xj.toArray.map(f => {
          val label = if (f.label == labels(i).toDouble) 1 else -1
          LabeledPoint(label.toDouble, f.features)
        })
        val lp = new LabelPair(labels(i), labels(j), p)

        x = p
        boundAlpha.clear
        alpha = new Array[Double](x.length)
        b = 0.0
        k = init()

        train()

        models += new Model(lp, alpha.clone, b)
      }
    }

    models.toArray
  }

  def loadData(): Array[LabeledPoint] = {
    val f = Source.fromFile(new File("D://tmp/heart_scale.csv"))
    val x = f.getLines.toArray.map(line => {
      var l = line.split(" ")
      val label = l(0).toDouble
      l = l.tail
      val data = l.map(v => v.split(":")(1).toDouble)
      LabeledPoint(label, data)
    })
    f.close
    x
  }

  def En(j: Int): Double = {
    var sum = 0.0
    for (i <- 0 until x.length) {
      sum += alpha(i) * x(i).label * k(i)(j)
    }

    sum + b - x(j).label

  }

  def init(): Array[Array[Double]] = {
    var r = new Array[Array[Double]](x.length)
    for (i <- 0 until x.length) {
      r(i) = new Array[Double](x.length)
      for (j <- 0 until x.length) {
        r(i)(j) = GuassionKernel(x(i), x(j))
        //                r(i)(j) = {
        //                  val va = x(i).features.zip(x(j).features)
        //                  val sum = va.map(t => {
        //                    (t._1 - t._2) * (t._1 - t._2)
        //                  })
        //                  math.exp(-1 * sum.reduce(_ + _) / 0.4) //Gaussian kernel
        //                }
        //        r(i)(j) = {
        //          val sum = 0.0;
        //          //System.out.println("x.length:" + x.length + "x[i].length" + x[i].length);
        //          val va = x(i).features.zip(x(j).features).map(t => t._1 * t._2).reduce(_ + _)
        //
        //          sum + va  //linear
        //        }
      }
    }
    r
  }

  def findMax(i: Int, Ei: Double): Int = {

    var max = 0.0
    var maxIndex = -1
    //val ite = boundAlpha.iterator
    //while (ite.hasNext) {
    for (j <- 0 until x.length) {
      //val j = ite.next
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

  def randomSelect(i: Int): Int = {
    var j = -1
    do {
      j = (Math.random * x.length).toInt
    } while (i == j);
    j
  }

  //SMO algorithm
  def train() {
    val maxPasses = 10
    var nowPass = 0

    alpha = alpha.map(f => 0.0)

    while (nowPass < maxPasses) {
      var num_changed_alphas = 0

      for (i <- 0 until x.length) {
        val Ei = En(i)
        val yi = x(i).label
        //KKT violattion condition
        if ((yi * Ei < -tol && alpha(i) < C) ||
          (yi * Ei > tol && alpha(i) > 0) ||
          (yi * Ei > (-tol) && (yi * Ei < tol) && (alpha(i) == 0.0 || alpha(i) == C))) {

          //select max |Ei- Ej|
          var j = if (boundAlpha.size > 0) findMax(i, Ei) else randomSelect(i)

          j = if (j < 0) randomSelect(i) else j
          val Ej = En(j)

          val oldAi = alpha(i)
          val oldAj = alpha(j)

          val yj = x(j).label

          //0< aj < C
          //ai + aj = E  //(Constant)
          var L = if (yi != yj) Math.max(0, alpha(j) - alpha(i)) else Math.max(0, alpha(i) + alpha(j) - C)
          val H = if (yi != yj) Math.min(C, C - alpha(i) + alpha(j)) else Math.min(C, alpha(j) + alpha(i))

          val eta = 2 * k(i)(j) - k(i)(i) - k(j)(j);

          val r = if (eta < 0) {
            update(i, j, Ei, Ej, oldAi, oldAj, L, H, eta)
          } else {
            System.out.println("eta out i= " + i)
            0
          }

          num_changed_alphas += r
        }
      }
      if (num_changed_alphas == 0) {
        nowPass += 1
      } else {
        nowPass = 0
      }
      System.out.println("iteration out")
    }

  }
  def update(i: Int, j: Int,
    Ei: Double, Ej: Double, oldAi: Double,
    oldAj: Double, L: Double, H: Double, eta: Double): Int = {
    val yj = x(j).label
    val yi = x(i).label
    //update aj
    alpha(j) = alpha(j) - yj * (Ei - Ej) / eta //公式(2)
    if (0 < alpha(j) && alpha(j) < C)
      boundAlpha.add(j)

    if (alpha(j) < L)
      alpha(j) = L
    else if (alpha(j) > H)
      alpha(j) = H

    //update ai
    if (Math.abs(alpha(j) - oldAj) >= 1e-5) {
      alpha(i) = alpha(i) + yi * yj * (oldAj - alpha(j))
      if (0 < alpha(i) && alpha(i) < C)
        boundAlpha.add(i)

      /** 计算b1， b2*/
      var b1 = b - Ei - yi * (alpha(i) - oldAi) * k(i)(i) - yj * (alpha(j) - oldAj) * k(i)(j)
      var b2 = b - Ej - yi * (alpha(i) - oldAi) * k(i)(j) - yj * (alpha(j) - oldAj) * k(j)(j)

      //update b
      if (0 < alpha(i) && alpha(i) < C)
        b = b1
      else if (0 < alpha(j) && alpha(j) < C)
        b = b2
      else
        b = (b1 + b2) / 2

      1
    } else {
      0
    }

  }

  def predict(): Double = {
    var probability = 0.0
    var correctCnt = 0
    var total = x.length

    for (i <- 0 until total) {
      //原来训练矩阵的维数（长度）
      val len = x.length;
      var sum = 0.0;
      for (j <- 0 until len) {
        sum += x(j).label * alpha(j) * k(j)(i);
      }
      sum += b;
      if ((sum > 0 && x(i).label > 0) || (sum < 0 && x(i).label < 0))
        correctCnt += 1
    }
    probability = correctCnt * 1.0 / total
    probability
  }

  def GuassionKernel(xj: LabeledPoint, xi: LabeledPoint): Double = {
    val sum = 0.0
    val zips = xj.features.zip(xi.features)
    val va = zips.map(t => (t._1 - t._2) * (t._1 - t._2))
    math.exp(-1 * va.reduce(_ + _) / 0.4)
  }

  def linearKernel(xj: LabeledPoint, xi: LabeledPoint): Double = {

    xi.features.zip(xj.features).
      map(t => t._1 * t._2).reduce(_ + _)

  }

  def polymonialKernel(xj: LabeledPoint, xi: LabeledPoint): Double = {
    val p = 3
    val sum = linearKernel(xj, xi)
    math.pow(sum + 1, p)
  }

  def rbfKernel(xj: LabeledPoint, xi: LabeledPoint): Double = {
    //e^-(gamma * <x-y, x-y>^2)
    val sum = 0.0
    val zips = xj.features.zip(xi.features)
    val va = zips.map(t => (t._1 - t._2) * (t._1 - t._2))

    val gama = 0.01
    math.exp(-gama * va.reduce(_ + _))

  }

  def predictMutilClass(test: Array[LabeledPoint], models: Array[Model]) {
    var probability = 0.0
    var correctCnt = 0
    val total = test.length

    for (i <- 0 until total) {
      val xi = test(i)

      val result = models.map(model => {
        //val model = models(m)
        val m_a = model.alpha
        val m_b = model.b
        //val m_k = model.k

        val m_x = model.forlabel.x
        val len = m_x.length;
        var sum = 0.0;
        for (j <- 0 until len) {
          sum += m_x(j).label * m_a(j) * GuassionKernel(m_x(j), xi);
        }
        sum += m_b
        val label = if (sum > 0) model.forlabel.l1 else model.forlabel.l2
        (label, 1)
      })
      val s = result.groupBy(t => t._1).map(f => { (f._1, f._2.size) })
      print(xi.label + " => ")
      var max = -1
      var index = "1"
      s.map(f => {
        //print(f._1+":"+f._2+" => ")
        if (f._2 > max) { max = f._2; index = f._1 }
      })
      print(index)

      if (index.toDouble == xi.label) { correctCnt += 1 }
      println()
    }
    println("correct:" + (correctCnt * 1.0 / total))
  }

  def main(args: Array[String]): Unit = {

    val models = trainMulti()

    val input = loadCsv()
    val p = predictMutilClass(input, models)

    //System.out.println(p)
  }

}