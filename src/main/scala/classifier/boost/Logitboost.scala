package classifier.boost

import classifier.Classifier
import classifier.Model
import core.Instances
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashMap
import classifier.tree.DSModel
import classifier.tree.DicisionStump
import core.Feature
import scala.collection.mutable.HashSet

//Additive Logistic Regression: a Statistical View of Boosting
//multiclass algorithm 6
class Logitboost(insts: Instances, M: Int) extends Classifier {

  def train(): Model = {
    val Z_MAX = 3.0

    val K = insts.classof.size
    val Ks = insts.classof.toArray

    val probs = Array.fill(insts.data.size)(Array.fill(K)(1.0 / K))
    val trainYs = Array.fill(insts.data.size)(Array.fill(K)(0.0))
    val trainFs = Array.fill(insts.data.size)(Array.fill(K)(0.0))
    val X = Array.fill(insts.data.size)(Array.fill(insts.attr)(""))
    val Y = Array.fill(insts.data.size)("")
    val W = Array.fill(insts.data.size)(0.0)

    for (i <- 0 until insts.data.size) {
      val inst = insts.data(i)
      X(i) = inst.features
      Y(i) = 1.0 + ""
      W(i) = 1.0
      val j = insts.classToInt(inst.target)
      trainYs(i)(j) = 1.0
    }
    val reg = new Instances(insts.numIdx, true)
    reg.read(X, Y)

    var Fkm = ArrayBuffer[HashMap[String, DSModel]]()

    for (m <- 0 until M) {
      //sum = y
      println("MM----------------MM")
      val fk = HashMap[String, DSModel]()
      for (iclass <- Ks.iterator) {
        val data = reg.data
        println("class:" + iclass)
        val weightsSumBefore = data.map(f => f.weight).sum
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

          inst.target = z + ""
          W(i) = inst.weight * w
          inst.weight = W(i)
        }

        val weightsSumAfter = reg.data.map(lf => lf.weight).sum
        val scale = weightsSumBefore / weightsSumAfter
        reg.data.map(lf => { lf.weight = lf.weight * scale })

        val ds = new DicisionStump(reg)

        //DecisionStump.printTree(nodes, nodes(1), 0)
        fk(iclass) = ds.train
      }

      Fkm += fk
      // Evaluate / increment trainFs from the classifier
      for (i <- 0 until trainFs.length) {
        val pred = Array.fill(K)(0.0)
        var predSum = 0.0
        for (k <- Ks.iterator) {
          val j = insts.classToInt(k)
          val model = fk(k)
          pred(j) = model.getRegValue(reg.data(i),
            !insts.numIdx.contains(model.node.iAttr), model.node.iAttr, model.node.split)
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

    LBModel(Fkm)
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
  def distributionForInstance(instance: Feature,
    Fkm: ArrayBuffer[HashMap[String, DSModel]],
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
        val model = Fkm(i)(k)
        pred(k) = model.getRegValue(instance,
          !insts.numIdx.contains(model.node.iAttr), model.node.iAttr, model.node.split)
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
  case class LBModel(Fkm: ArrayBuffer[HashMap[String, DSModel]]) extends Model {

    def predict(test: Instances): Double = {
      var r = 0.0
      val numIdx = test.numIdx
      val Ks = test.classof.toArray
      test.data.map(lf => {
        var dist = distributionForInstance(lf, Fkm, Ks.size, Ks, numIdx)
        //normalize
        val sum = dist.values.sum
        dist = dist.map(f => { (f._1, f._2 / sum) })

        // find the max probability
        val rst = dist.toArray.sortBy(f => f._2).reverse
        //label of the max probability
        val label = rst(0)._1
        //if hit 
        if (label.equalsIgnoreCase(lf.target)) r += 1.0
        println(lf.target + "=>" + dist)
      })
      r / test.data.size
    }
  }
}

object Logitboost {

  def main(args: Array[String]): Unit = {

    var numIdx = new HashSet[Int]
    numIdx.+=(0)
    numIdx.+=(1)
    numIdx.+=(2)
    numIdx.+=(3)
        numIdx.+=(5)
        numIdx.+=(7)
        numIdx.+=(8)
        numIdx.+=(10)

    val insts = new Instances(numIdx)
    insts.read("E:/books/spark/ml/decisionTree/labor.csv")
    //    insts.all2Normal
    val (trainset, testset) = insts.stratify()

    val t = new Logitboost(trainset, 21)
    val model = t.train()

    val accu = model.predict(testset)
    println(accu);
  }
}