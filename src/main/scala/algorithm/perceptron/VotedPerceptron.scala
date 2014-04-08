package algorithm.perceptron

import scala.collection.mutable.HashSet
import algorithm.Instances
import scala.collection.mutable.ArrayBuffer
import algorithm.LabeledFeature
import scala.collection.mutable.HashMap

////http://curtis.ml.cmu.edu/w/courses/index.php/Voted_Perceptron
object VotedPerceptron {
  case class Model(vk: ArrayBuffer[Array[Double]], ck: ArrayBuffer[Int], pair: (String, String))

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

        val vk = new ArrayBuffer[Array[Double]]()
        val ck = ArrayBuffer[Int]()
        val model = train(traindata, vk, ck, pair, rate, T)
        //val model = trainByGram(traindata, pair, rate, T)
        val matrix = predict(model, testdata, pair)

        //val matrix = predictByGram(model, testdata, pair)
        val a = matrix.map(f => f._2.values.sum).sum
        r += (matrix(pair._1).getOrElse(pair._1, 0) + matrix(pair._2).getOrElse(pair._2, 0)) * 1.0 / a
      }
      avg += (r / fold)
      val vk = new ArrayBuffer[Array[Double]]()
      val ck = ArrayBuffer[Int]()
      val alldata = classA.++(classB)
      shuffle(alldata)
      val fmodel = train(alldata, vk, ck, pair, rate, T)
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
  def train(data: ArrayBuffer[LabeledFeature],
    vk: ArrayBuffer[Array[Double]],
    ck: ArrayBuffer[Int],
    pair: (String, String), rate: Double, T: Int): Model = {
    //var b = ck
    var err = data.size
    var j = T
    val vt = Array.fill(data(0).features.length)(0.0)
    vk.+=(vt)
    var ct = 0
    ck += (ct)
    var k = 0
    while ( /**err > 0 && */ j > 0) {
      //println(err)
      err = 0
      data.map(f => {
        val yi = if (f.label.equalsIgnoreCase(pair._1)) 1 else -1
        val yx = (vk(k).zip(f.features).map(t => t._1 * t._2.toDouble).sum)
        val y_ = if (yx <= 0) -1 else 1
        if (yi == y_) {
          ck(k) = ck(k) + 1
        } else {
          val vt1 = vk(k).zip(f.features).map(t => t._1 + yi * t._2.toDouble)
          vk.+=(vt1)
          ck += (1)
          k = k + 1
          err = err + 1
        }
      })
      j -= 1
    }

    Model(vk, ck, pair)
  }

  def predict(model: Model,
    test: ArrayBuffer[LabeledFeature],
    pair: (String, String)): HashMap[String, HashMap[String, Int]] = {
    val vk = model.vk
    val ck = model.ck
    //println(pair)
    val matrix = new HashMap[String, HashMap[String, Int]]
    matrix(pair._1) = new HashMap[String, Int]
    matrix(pair._2) = new HashMap[String, Int]
    test.map(f => {
      val label = f.label
      //val yi = if (f.label.equalsIgnoreCase(pair._1)) 1 else -1
      //val yx = (w.zip(f.features).map(t => t._1 * t._2.toDouble).sum + b)
      var yx = 0.0
      for (k <- 0 until vk.size) {
        val tmp = vk(k).zip(f.features).map(t => t._1 * t._2.toDouble).sum
        yx += ck(k) * (if (tmp <= 0) -1 else 1)
      }
      val l = if (yx <= 0) -1 else 1
      val rlabel = if (1 == l) pair._1 else pair._2
      matrix(label)(rlabel) = matrix(label).getOrElse(rlabel, 0) + 1
    })
    matrix
  }

  def predictMulti(models: ArrayBuffer[Model],
    f: LabeledFeature): String = {
    val matrix = new HashMap[String, Int]

    for (model <- models) {
      val vk = model.vk
      val ck = model.ck
      val pair = model.pair
      //if(!matrix.contains(pair._1))matrix(pair._1) = new HashMap[String, Int]
      //if(!matrix.contains(pair._2))matrix(pair._2) = new HashMap[String, Int]
      //matrix(pair._2) = new HashMap[String, Int]
      //test.map(f => {
      //val label = f.label
      //val yi = if (f.label.equalsIgnoreCase(pair._1)) 1 else -1
      //val yx = (w.zip(f.features).map(t => t._1 * t._2.toDouble).sum + b)
      var yx = 0.0
      for (k <- 0 until vk.size) {
        yx += ck(k) * vk(k).zip(f.features).map(t => t._1 * t._2.toDouble).sum
      }
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

    classifier(insts, 10, 0.1, 10)

  }
}