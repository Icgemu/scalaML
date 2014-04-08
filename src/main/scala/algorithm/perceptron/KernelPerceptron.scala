package algorithm.perceptron

import algorithm.Instances
import algorithm.Kernel._
import scala.collection.mutable.HashMap
import algorithm.LabeledFeature
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashSet

object KernelPerceptron {
  case class Model(b: Double, w: Array[Double], pair: (String, String), y: Array[(Int, Int)])

  def makePair(cls: HashSet[String]): HashSet[(String, String)] = {
    val pair = new HashSet[(String, String)]
    cls.map(f => {
      cls.map(t => {
        if (t < f) pair.+=((t, f)) else pair.+=((f, t))
      })
    })
    val f = pair.filterNot(t => { t._1.equalsIgnoreCase(t._2) })
    f
  }

  def Kernel(data: ArrayBuffer[LabeledFeature]): Array[Array[Double]] = {

    val gram = Array.fill(data.size)(Array.fill(data.size)(0.0))

    for (i <- 0 until data.size) {
      for (j <- 0 until data.size) {
        gram(i)(j) = GuassionKernel(data(i), data(j))
      }
    }
    gram

  }

  var gram: Array[Array[Double]] = null
  //原始形式
  def classifier(insts: Instances, T: Int, rate: Double, fold: Int) {
    val numClass = insts.numClass
    val classData = new HashMap[String, ArrayBuffer[LabeledFeature]]

    gram = Kernel(insts.data)

    for (cls <- insts.classof) {
      if (!classData.contains(cls)) classData(cls) = new ArrayBuffer[LabeledFeature]
      classData(cls) = insts.data.filter(p => p.label.equalsIgnoreCase(cls))
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

      if (!matrixs.contains(p._1)) {
        matrixs(p._1) = new HashMap[String, Int]
      }
      matrixs(p._1) += ((p._2, 0))

    }

    val models = ArrayBuffer[Model]()
    //val matrixs = new HashMap[String,HashMap[String,Int]]
    var avg = 0.0
    for (pair <- clsPair) {
      //println(pair)
      val classA = classData(pair._1)
      val classB = classData(pair._2)

      val foldA = makeFold(classA, fold)
      val foldB = makeFold(classB, fold)
      //val data = classData(pair._1).++()
      var r = 0.0
      for (i <- 0 until fold) {
        val traindata = mergeFold(foldA, i).++(mergeFold(foldB, i))
        val testdata = foldA(i).++=(foldB(i))
        shuffle(traindata)
        val w = Array.fill(insts.attr)(0.0)
        val bias = 0.0
        // val model = train(traindata, w, bias, pair, rate,T)
        val model = trainByGram(traindata, pair, rate, T)
        //val matrix = predict(model, testdata, pair)
        val matrix = predictByGram(model, testdata, pair)
        val a = matrix.map(f => f._2.values.sum).sum
        r += (matrix(pair._1).getOrElse(pair._1, 0) + matrix(pair._2).getOrElse(pair._2, 0)) * 1.0 / a
      }
      avg += (r / fold)
      val w = Array.fill(insts.attr)(0.0)
      val bias = 0.0
      val alldata = classA.++(classB)
      shuffle(alldata)
      //val fmodel = train(alldata,w,bias,pair,rate,T)
      val fmodel = trainByGram(alldata, pair, rate, T)
      models += fmodel
    }
    val data = insts.data
    data.map(f => {
      //predictMulti(models,f)
      predictMultiByGram(models, f)
    })

    print("r=" + avg / clsPair.size)
  }
  def shuffle(data: ArrayBuffer[LabeledFeature]) {
    val size = data.size
    for (i <- 0 until size) {
      val r1 = (size * math.random).toInt
      val r2 = (size * math.random).toInt
      // val v = data(r)
      val r1f = data(r1)
      val r2f = data(r2)
      data(r1) = r2f
      data(r2) = r1f
      //data.+=(v)
    }
  }

  def trainByGram(data: ArrayBuffer[LabeledFeature],
    //gram: Array[Array[Double]],
    ///bias: Double,
    pair: (String, String), rate: Double, T: Int): Model = {
    var b = 0.0
    var err = data.size
    var j = T
    val alpha = Array.fill(data.size)(0.0)
    val y: Array[(Int, Int)] = Array.fill(data.size)(null)
    while (err > 0 /**&& j>0*/ ) {
      //println(err)
      err = 0
      //data.map(f => {
      for (m <- 0 until data.size) {
        val f = data(m)
        val i = f.i
        val yi = if (f.label.equalsIgnoreCase(pair._1)) 1 else -1
        var sum = 0.0
        for (n <- 0 until data.size) {
          val f2 = data(n)
          val j = f2.i
          val yj = if (f2.label.equalsIgnoreCase(pair._1)) 1 else -1
          sum += alpha(n) * yj * gram(i)(j)
        }
        val yx = yi * (sum + b)
        y(m) = (yi, i)
        if (yx <= 0) {
          alpha(m) = alpha(m) + rate
          b = b + rate * yi
          err += 1
        }
      }
      //})
      j -= 1
    }

    Model(b, alpha, pair, y)
    //Model(bias,w,pair,null)
  }
  def predictByGram(model: Model,
    test: ArrayBuffer[LabeledFeature],
    //y: Array[Int],
    pair: (String, String)): HashMap[String, HashMap[String, Int]] = {
    val b = model.b
    val alpha = model.w
    val y = model.y
    //val gram = model.gram
    //println(pair)
    val matrix = new HashMap[String, HashMap[String, Int]]
    matrix(pair._1) = new HashMap[String, Int]
    matrix(pair._2) = new HashMap[String, Int]
    //test.map(f => {
    for (m <- 0 until test.size) {

      val f = test(m)
      val i = f.i
      val label = f.label
      //val yi = if (f.label.equalsIgnoreCase(pair._1)) 1 else -1
      var sum = 0.0
      for (n <- 0 until alpha.size) {
        //val f2 = traindata(n)
        //val j = f2.i
        //val yj = if (f2.label.equalsIgnoreCase(pair._1)) 1 else -1
        sum += alpha(n) * y(n)._1 * gram(i)(y(n)._2)
      }
      val yx = (sum + b)
      val l = if (yx <= 0) -1 else 1
      val rlabel = if (1 == l) pair._1 else pair._2
      matrix(label)(rlabel) = matrix(label).getOrElse(rlabel, 0) + 1
      //})
    }
    matrix
  }

  def predictMultiByGram(models: ArrayBuffer[Model],
    f: LabeledFeature): String = {
    val matrix = new HashMap[String, Int]

    for (model <- models) {
      val b = model.b
      val alpha = model.w
      val pair = model.pair
      val y = model.y
      //if(!matrix.contains(pair._1))matrix(pair._1) = new HashMap[String, Int]
      //if(!matrix.contains(pair._2))matrix(pair._2) = new HashMap[String, Int]
      //matrix(pair._2) = new HashMap[String, Int]
      //test.map(f => {
      //val label = f.label
      //val yi = if (f.label.equalsIgnoreCase(pair._1)) 1 else -1
      var sum = 0.0
      for (n <- 0 until alpha.size) {
        //val f2 = traindata(n)
        //val j = f2.i
        //val yj = if (f2.label.equalsIgnoreCase(pair._1)) 1 else -1
        sum += alpha(n) * y(n)._1 * gram(f.i)(y(n)._2)
      }
      val yx = (sum + b)
      // val yx =  (w.zip(f.features).map(t => t._1 * t._2.toDouble).sum + b)
      val l = if (yx <= 0) -1 else 1
      val rlabel = if (1 == l) pair._1 else pair._2
      matrix(rlabel) = matrix.getOrElse(rlabel, 0) + 1
      //})
    }
    val arr = matrix.toArray.sortBy(f => { f._2 }).reverse
    println(f.label + "=>" + matrix)
    arr(0)._1
  }
  def mergeFold(f: Array[ArrayBuffer[LabeledFeature]],
    i: Int): ArrayBuffer[LabeledFeature] = {
    val merge = new ArrayBuffer[LabeledFeature]()
    for (j <- 0 until f.size) {
      if (i != j) { merge.++=(f(j)) }
    }
    merge
  }
  def makeFold(data: ArrayBuffer[LabeledFeature],
    fold: Int): Array[ArrayBuffer[LabeledFeature]] = {

    val f = Array.fill(fold)(new ArrayBuffer[LabeledFeature])
    val r = if (data.size % fold == 0) data.size / fold else data.size / fold + 1
    for (i <- 0 until r) {
      for (j <- 0 until fold) {
        if ((i + j) < data.size)
          f(j).+=(data(i + j))
      }
    }
    f
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

    classifier(insts, 20, 0.1, 10)

  }
}