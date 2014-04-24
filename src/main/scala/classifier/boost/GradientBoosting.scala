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

//GBDT-Gradient Boosting Decision Tree

//Greedy Function Approximation A Gradient Boosting Machine:Algorithm 6:Lk-TreeBoost
class GradientBoosting(insts: Instances, M: Int) extends Classifier {

  def train(): Model = {

    val K = insts.classof.size
    val Ks = insts.classof.toArray

    //models
    var Fkm = ArrayBuffer[HashMap[String, DSModel]]()
    //init prob  
    var Fk0 = HashMap[String, Double]()
    Ks.map(f => Fk0(f) = math.random)
    for (i <- 0 until M) {
      println("MM----------------MM")
      val fk = HashMap[String, DSModel]()

      for (iclass <- Ks.iterator) {

        val X = Array.fill(insts.data.size)(Array.fill(insts.attr)(""))
        val Y = Array.fill(insts.data.size)("")
        
        // SGD can use here use subset of training data 
        // recommend sample rate:  0.5=<r<=0.8
        for (i <- 0 until insts.data.size) {
          val inst = insts.data(i)

          X(i) = inst.features

          //probability of current inst fit to this class
          val prob = probk(inst, Ks, iclass, Fk0, Fkm, insts.numIdx)

          //current residual gradient according to Fk,m-1
          //if gk >0,it mean porb should increase   
          //if gk <0,it mean porb should decrease
          //if gk close 0,it mean prob is corretly predict...

          val gk = (if (inst.target.equalsIgnoreCase(iclass)) 1.0 else 0.0) - prob
          //sum += gk * gk
          Y(i) = gk + ""
        }
        //println("Log=" + sum)

        //train a regression model according to new residual gradient 
        val reg = new Instances(insts.numIdx, true)
        reg.read(X, Y)
        val ds = new DicisionStump(reg)

        fk(iclass) = ds.train
      }
      Fkm += fk
    }
    GDBTModel(Fk0, Fkm)
  }

  def probk(
    x: Feature, //feature to predict
    cls: Array[String],//all label class
    expectedClass: String,//expected label
    Fk0: HashMap[String, Double],//init probability
    Fkm: ArrayBuffer[HashMap[String, DSModel]],//current models
    numIdx: HashSet[Int]//index for numerical attr
    ): Double = {
    var fork = 0.0
    var sum = 0.0
    for (iclss <- cls.toIterator) {

      val t = (Fkm.map(f => {
        val model = f(iclss)
        val rjkm = model.getRegValue(x,
          !numIdx.contains(model.node.iAttr), model.node.iAttr, model.node.split)
        ((cls.size - 1) * 1.0 / cls.size) * rjkm
      }).sum + Fk0(iclss))
      sum += math.exp(t)
      if (iclss.equalsIgnoreCase(expectedClass)) {
        fork = math.exp(t)
      }
    }
    // logistic transformation
    fork / sum
  }

  case class GDBTModel(Fk0: HashMap[String, Double],
    Fkm: ArrayBuffer[HashMap[String, DSModel]]) extends Model {

    def predict(test: Instances): Double = {
      val numIdx = test.numIdx
      val Ks = test.classof.toArray
      var r = 0.0
      test.data.map(lf => {
        var dist = HashMap[String, Double]()
        for (k <- Ks.toIterator) {
          dist(k) = probk(lf, Ks, k, Fk0, Fkm, numIdx)
        }
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
        //println(lf.target)
      })
      r / test.data.size
    }
  }
}

object GradientBoosting {

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
    //    insts.all2Normal
    val (trainset, testset) = insts.stratify()

    val t = new GradientBoosting(trainset, 21)
    val model = t.train()

    val accu = model.predict(testset)
    println(accu);
  }
}