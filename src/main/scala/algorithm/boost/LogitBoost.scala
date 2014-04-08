package algorithm.boost

import algorithm.Instances
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashMap
import algorithm.tree.DecisionStump.Node
import algorithm.RegInstances
import algorithm.tree.DecisionStump
import algorithm.LabeledFeature
import scala.collection.mutable.HashSet

object LogitBoost {

  val Z_MAX = 3.0

  def classifier(insts: Instances, M: Int, J: Int) = {
    val K = insts.classof.size
    val Ks = insts.classof.toArray

    val probs = Array.fill(insts.data.size)(Array.fill(K)(1.0 / K))
    val trainYs = Array.fill(insts.data.size)(Array.fill(K)(0.0))
    val trainFs = Array.fill(insts.data.size)(Array.fill(K)(0.0))
    val X = Array.fill(insts.data.size)(Array.fill(insts.attr)(""))
    val Y = Array.fill(insts.data.size)(0.0)
    val W = Array.fill(insts.data.size)(0.0)

    for (i <- 0 until insts.data.size) {
      val inst = insts.data(i)
      X(i) = inst.features
      Y(i) = 1.0
      W(i) = 1.0
      val j = insts.classToInt(inst.label)
      trainYs(i)(j) = 1.0
    }
    val reg = new RegInstances(insts.numIdx)
    reg.read(X, Y)

    var Fkm = ArrayBuffer[HashMap[String, HashMap[Int, Node]]]()

    for (m <- 0 until M) {
      //sum = y
      println("MM----------------MM")
      val tmp = HashMap[String, HashMap[Int, Node]]()
      for (iclass <- Ks.iterator) {
        val data = reg.data
        println("class:" + iclass)
        val wbefore = data.map(f => f.weight).sum
        for (i <- 0 until data.size) {
          val j = insts.classToInt(iclass)
          val inst = data(i)
          val p = probs(i)(j)
          var z, actual = trainYs(i)(j)
          if (actual == 1.0) {
            z = 1.0 / p
            if (z > Z_MAX) { // threshold
              z = Z_MAX;
            }
          } else {
            z = -1.0 / (1.0 - p);
            if (z < -Z_MAX) { // threshold
              z = -Z_MAX;
            }
          }
          val w = (actual - p) / z

          inst.value = z
          W(i) = inst.weight * w
          inst.weight = W(i)
        }

        val wafter = reg.data.map(lf => lf.weight).sum
        val scale = wbefore / wafter
        reg.data.map(lf => { lf.weight = lf.weight * scale })

        val nodes = DecisionStump.classifier(reg, 2)
        DecisionStump.printTree(nodes, nodes(1), 0)
        tmp(iclass) = nodes
      }

      Fkm += tmp
      // Evaluate / increment trainFs from the classifier
      for (i <- 0 until trainFs.length) {
        val pred = Array.fill(K)(0.0)
        var predSum = 0.0
        for (k <- Ks.iterator) {
          val j = insts.classToInt(k)
          pred(j) = 1.0 * DecisionStump.instanceFor(tmp(k), 1, reg.data(i).features, reg.numIdx)
          predSum += pred(j)
        }
        predSum /= K
        for (k <- Ks.iterator) {
          val j = insts.classToInt(k)
          trainFs(i)(j) += (pred(j) - predSum) * (K - 1) / K
        }
      }

      // Compute the current probability estimates
      for (i <- 0 until trainYs.length) {
        probs(i) = probst(trainFs(i))
      }
    }

    test(insts, Fkm)
  }

  def probst(Fs: Array[Double]): Array[Double] = {
    var maxF = -Double.MaxValue
    for (i <- 0 until Fs.length) {
      if (Fs(i) > maxF) {
        maxF = Fs(i)
      }
    }
    var sum = 0.0
    var probs = Array.fill(Fs.length)(0.0)
    for (i <- 0 until Fs.length) {
      probs(i) = Math.exp(Fs(i) - maxF)
      sum += probs(i)
    }
    //Utils.normalize(probs, sum);
    probs = probs.map(f => f / sum)
    probs
  }
  def distributionForInstance(instance: LabeledFeature,
    Fkm: ArrayBuffer[HashMap[String, HashMap[Int, Node]]],
    m_NumClasses: Int,
    Ks: Array[String],
    numIdx: HashSet[Int]): HashMap[String, Double] = {

    // instance = (Instance)instance.copy();
    //instance.setDataset(m_NumericClassData);
    val pred = HashMap[String, Double]()
    val Fs = HashMap[String, Double]()
    for (i <- 0 until Fkm.size) {
      var predSum = 0.0
      for (k <- Ks.toIterator) {

        pred(k) = 1.0 * DecisionStump.instanceFor(Fkm(i)(k), 1, instance.features, numIdx)
        predSum += pred(k)
      }
      predSum /= m_NumClasses
      for (j <- Ks.toIterator) {
        Fs(j) = Fs.getOrElse(j, 0.0) + (pred.getOrElse(j, 0.0) - predSum) * (m_NumClasses - 1) / m_NumClasses
      }
    }
    //val s = Fs.values.sum
    val p = probst(Fs.values.toArray)
    var i = 0
    Fs.map(f => { val t = (f._1, p(i)); i += 1; t })
  }

  def test(insts: Instances,
    Fkm: ArrayBuffer[HashMap[String, HashMap[Int, Node]]]) {
    val numIdx = insts.numIdx
    val Ks = insts.classof.toArray
    insts.data.map(lf => {
      val p = distributionForInstance(lf, Fkm, Ks.size, Ks, numIdx)
      for (k <- Ks.toIterator) {
        print(k + "=" + p(k) + ",")
      }
      println(lf.label)
    })

  }
  def main(args: Array[String]): Unit = {

    var numIdx = new HashSet[Int]
    numIdx.+=(0)
    numIdx.+=(1)
    numIdx.+=(2)
    numIdx.+=(3)
    //    numIdx.+=(4)
    numIdx.+=(5)
    numIdx.+=(8)
    numIdx.+=(10)
    val insts = new Instances(numIdx)
    insts.read("E:/books/spark/ml/decisionTree/labor.csv")

    classifier(insts, 10, 2)
  }

}