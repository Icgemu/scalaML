package algorithm.regression

import algorithm.Instances
import algorithm.LabeledFeature
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashSet
import scala.collection.mutable.HashMap
import algorithm.matrix.MatrixDecomposition._

//非正则化下的Logistic Regression with SGD，容易 over fitting
object LogisticRegression {

  case class Model(w: Array[Double], pair: (String, String))

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

  //var gram: Array[Array[Double]] = null

  //原始形式
  def classifier(insts: Instances, T: Int, rate: Double, fold: Int) {
    val numClass = insts.numClass
    val classData = new HashMap[String, ArrayBuffer[LabeledFeature]]

    //gram = Kernel(insts.data)

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
        //        val w = Array.fill(insts.attr)(0.0)
        //        val bias = 0.0
        val model = train(traindata, pair, T, rate)
        //val model = trainByGram(traindata, pair, rate, T)
        val matrix = predict(model, testdata, pair)
        //val matrix = predictByGram(model, testdata, pair)
        val a = matrix.map(f => f._2.values.sum).sum
        r += (matrix(pair._1).getOrElse(pair._1, 0) + matrix(pair._2).getOrElse(pair._2, 0)) * 1.0 / a
      }
      avg += (r / fold)
      //      val w = Array.fill(insts.attr)(0.0)
      //      val bias = 0.0
      val alldata = classA.++(classB)
      shuffle(alldata)
      val fmodel = train(alldata, pair, T, rate)
      //val fmodel = trainByGram(alldata, pair, rate, T)
      models += fmodel
    }
    val data = insts.data
    data.map(f => {
      predictMulti(models, f)
      //predictMultiByGram(models, f)
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
  def train(input: ArrayBuffer[LabeledFeature],
    pair: (String, String), T: Int, rate: Double): Model = {

    var featureLen = input(0).features.length
    var initWeights = Array.fill[Double](featureLen + 1)(math.random)

    var data = input.map(p => {
      var fs = p.features.map(f => f.toDouble)
      val arr = Array(1.0, fs: _*)
      //val l = if(p.label.equals(pair._1)) 1+"" else 0+""
      new LabeledFeature(p.i, p.label, arr.map(f => f.toString()), p.weight)
    })

    assert(data(0).features.length == initWeights.length)
    //    for (i <- 0 to 1000) {
    //      initWeights = gradientDescent(data, initWeights)
    //    }
    //initWeights = SGD_miniBacth(data, initWeights, 0.6, 1000)
    //initWeights = SGD(data, initWeights, 1000)
    initWeights = gradientDescent(data, initWeights, pair, T, rate)
    //initWeights = Bfgs(data, initWeights, pair, T, rate)
    var w: Array[Double] = Array[Double](initWeights: _*)
    //System.out.println(w.map(d => print(d + ":")))
    Model(w, pair)
  }

  //Gradient Descent Algorithm
  def gradientDescent(data: ArrayBuffer[LabeledFeature],
    initWeights: Array[Double], pair: (String, String), iteNum: Int, rate: Double): Array[Double] = {

    val count = data.size.toInt
    for (j <- 0 until iteNum) {
      val weights = data.map(f => {
        //val label = f.label.toDouble
        val label = if (f.label.equals(pair._1)) 1 else 0
        val point = f.features.map(f => f.toDouble)

        var sumWeights = new Array[Double](initWeights.length)
        for (i <- 0 until initWeights.length) {
          val r = (h(point, initWeights) - label) * point(i)
          sumWeights(i) = r
        }
        sumWeights
      }).reduce((a, b) => {
        val t = a.zip(b)
        val r = t.map(t => t._1 + t._2)
        r
      })
      for (i <- 0 until initWeights.length) {
        initWeights(i) = initWeights(i) - rate * (1.0 / count.toDouble) * weights(i)
      }
    }
    initWeights
  }

  def Bfgs(data: ArrayBuffer[LabeledFeature],
    initWeights: Array[Double], pair: (String, String), iteNum: Int, rate: Double): Array[Double] = {

    println("-------------")
    var B = randomorthoMatrix(initWeights.size)
    var gxk = data.map(f => {
      val lf = f.features.map(ff => ff.toDouble)
      val label = if (f.label.equals(pair._1)) 1 else 0
      val hw = h(lf, initWeights)
      val gradient = Array.fill(lf.size)(0.0)
      for (i <- 0 until lf.size) {
        gradient(i) = (label - hw) * lf(i)
      }
      gradient
    }).reduce((t1, t2) => { t1.zip(t2).map(t => t._1 + t._2) }).map(t => t / data.size)
    var isCov = false
    var nol = math.sqrt(gxk.map(f => f * f).sum)
    println(nol)
    while (!isCov) {
      val gk = Array.fill(1)(gxk)
      val pk = transpose(dot(reverse(B), multi(transpose(gk), -1.0)))(0)
      val weights = Array.fill(initWeights.size)(0.0)
      for (i <- 0 until pk.size) {
        weights(i) = initWeights(i) + rate * pk(i)
      }
      val gxkplus1 = data.map(f => {
        val lf = f.features.map(ff => ff.toDouble)
        val label = if (f.label.equals(pair._1)) 1 else 0
        val hw = h(lf, weights)
        val gradient = Array.fill(lf.size)(0.0)
        for (i <- 0 until lf.size) {
          gradient(i) = (label - hw) * lf(i)
        }
        gradient
      }).reduce((t1, t2) => { t1.zip(t2).map(t => t._1 + t._2) })
        .map(t => t / data.size)
      val nola = math.sqrt(gxkplus1.map(f => f * f).sum)
       println(nola)
       val cov = nol - nola
      if ( cov>=0 && cov < 1e-3) {
        isCov = true
      } else {
        
        val yk = transpose(Array.fill(1)(gxkplus1.zip(gxk).map(t => { t._1 - t._2 })))
        val gak = transpose(Array.fill(1)(weights.zip(initWeights).map(t => { t._1 - t._2 })))
        val v1 = dot(yk, transpose(yk))
        val v2 = dot(transpose(yk), gak)
        val s1 = dot(dot(dot(B, gak), transpose(gak)), B)
        val s2 = dot(dot(transpose(gak), B), gak)
        B = plus(B, plus(multi(v1, 1 / v2(0)(0)), multi(s1, -1.0 / s2(0)(0))))

        for (i <- 0 until weights.size) {
          initWeights(i) = weights(i)
        }
        gxk = gxkplus1
        
        nol = nola
      }
    }
    initWeights
  }

  //Stochastic Gradient Descent Algorithm
  // it seem can not parallel in spark
  def SGD(data: ArrayBuffer[LabeledFeature],
    initWeights: Array[Double], pair: (String, String), iteNum: Int): Array[Double] = {

    for (ite <- 0 until iteNum) {
      //var i = 0
      //var sumWeights = Array[Double](initWeights.length)

      data.map(f => {

        //val label = f.label.toDouble
        val label = if (f.label.equals(pair._1)) 1 else 0
        val point = f.features.map(f => f.toDouble)

        //var weights = h(point,initWeights)
        //var sumWeights = Array[Double](initWeights.length)
        for (i <- 0 until initWeights.length) {
          val r = (h(point, initWeights) - label) * point(i)
          val th = initWeights(i) - 0.01 * r
          initWeights(i) = th
        }

      })
    }
    //    weights
    initWeights
  }
  def sample(data: ArrayBuffer[LabeledFeature], fraction: Double): ArrayBuffer[LabeledFeature] = {
    val size = (data.size * fraction).toInt
    val arr = new ArrayBuffer[LabeledFeature]

    for (i <- 0 until size) {
      val j = data.size * math.random.toInt
      arr.+=:(data(j))
    }
    arr
  }
  //Mini-batch-Stochastic Gradient Descent Algorithm
  def SGD_miniBacth(data: ArrayBuffer[LabeledFeature],
    initWeights: Array[Double], pair: (String, String), fraction: Double, IteMun: Int, rate: Double): Array[Double] = {

    //val loop = (data.count / size).toInt;
    //var weights = Array[Double](initWeights.length)
    for (j <- 0 until IteMun) {
      var sampleData = sample(data, fraction)
      val size = sampleData.size;
      if (size > 0) {
        //for (i <- 0 until initWeights.length) {

        val sum = sampleData.map(f => {

          //val label = f.label.toDouble
          val label = if (f.label.equals(pair._1)) 1 else 0
          val point = f.features.map(f => f.toDouble)
          var t = Array.fill(initWeights.length)(0.0)
          for (i <- 0 until t.length) {
            val r = (h(point, initWeights) - label) * point(i)
            t(i) = r
          }
          t
        }).reduce((a, b) => {
          val z = a.zip(b)
          val r = z.map(t => t._1 + t._2)
          r
        })
        for (i <- 0 until initWeights.length) {
          initWeights(i) = initWeights(i) - rate * sum(i) / size
        }
      }
    }
    initWeights
  }

  def h(xi: Array[Double], q: Array[Double]): Double = {
    val v = -1 * xi.zip(q).map((t: (Double, Double)) => t._1 * t._2).sum
    val a = 1.0 / (1.0 + math.exp(v))
    a
  }

  def predict(x: Array[Double], w: Array[Double]): Double = {

    val r = h(x, w)
    System.out.println(r)
    r
  }
  def predict(model: Model,
    test: ArrayBuffer[LabeledFeature],
    pair: (String, String)): HashMap[String, HashMap[String, Int]] = {
    //val b = model.b
    val w = model.w
    //println(pair)
    var data = test.map(p => {
      var fs = p.features.map(f => f.toDouble)
      val arr = Array(1.0, fs: _*)
      //val l = if(p.label.equals(pair._1)) 1+"" else 0+""
      new LabeledFeature(p.i, p.label, arr.map(f => f.toString()), p.weight)
    })
    val matrix = new HashMap[String, HashMap[String, Int]]
    matrix(pair._1) = new HashMap[String, Int]
    matrix(pair._2) = new HashMap[String, Int]
    data.map(f => {
      val label = f.label
      //val yi = if (f.label.equalsIgnoreCase(pair._1)) 1 else -1
      val point = f.features.map(f => f.toDouble)
      val yx = h(point, w)
      val l = if (yx <= 0.5) 0 else 1
      val rlabel = if (1 == l) pair._1 else pair._2
      matrix(label)(rlabel) = matrix(label).getOrElse(rlabel, 0) + 1
    })
    matrix
  }

  def predictMulti(models: ArrayBuffer[Model],
    f: LabeledFeature): String = {
    val matrix = new HashMap[String, Int]

    for (model <- models) {
      //val b = model.b
      val w = model.w
      val pair = model.pair
      //if(!matrix.contains(pair._1))matrix(pair._1) = new HashMap[String, Int]
      //if(!matrix.contains(pair._2))matrix(pair._2) = new HashMap[String, Int]
      //matrix(pair._2) = new HashMap[String, Int]
      //test.map(f => {
      //val label = f.label
      //val yi = if (f.label.equalsIgnoreCase(pair._1)) 1 else -1

      val point = f.features.map(f => f.toDouble)
      val yx = h(Array(1.0, point: _*), w)
      val l = if (yx <= 0.5) 0 else 1

      val rlabel = if (1 == l) pair._1 else pair._2
      matrix(rlabel) = matrix.getOrElse(rlabel, 0) + 1
      //})
    }
    val arr = matrix.toArray.sortBy(f => { f._2 }).reverse
    println(f.label + "=>" + matrix)
    arr(0)._1
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

    classifier(insts, 100, 0.09, 10)
  }

}