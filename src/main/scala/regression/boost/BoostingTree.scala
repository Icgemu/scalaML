package regression.boost

import regression.Regression
import regression.Model
import core.Instances
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashMap
import classifier.tree.DSModel
import classifier.tree.DicisionStump
import core.Feature
import scala.collection.mutable.HashSet
import regression.tree.CARTRegression
import regression.tree.CARTRegModel
//MART-Multiple Additive Regression Tree
//Greedy Function Approximation A Gradient Boosting Machine:Algorithm 3:LAD-TreeBoost
class BoostingTree(insts: Instances, M: Int) extends Regression {

  def train(): Model = {

    //val all = insts.data.map(f=>f.weight).sum
    val target = insts.data.map(f=>f.target.toDouble)
    
    var Fkm = ArrayBuffer[CARTRegModel]()
    var is = true
    var y = Double.MaxValue
    while (is) {
      //sum = y
      println("MM----------------MM")
      val X = Array.fill(insts.data.size)(Array.fill(insts.attr)(""))
      val Y = Array.fill(insts.data.size)("")
      var sum = 0.0
      for (i <- 0 until insts.data.size) {
        val inst = insts.data(i)
        X(i) = inst.features
        
        val c = pk(inst, Fkm)
        val yik = target(i) - c
        
        sum += yik * yik
        Y(i) = yik + ""
      }
      println("Log=" + sum)
      if (y - sum < 100) {
        is = false
      } else {
        y = sum
      }
      val reg = new Instances(insts.numIdx, true)
      reg.read(X, Y)

      //val ds = new DicisionStump(reg)
      val cart = new CARTRegression(reg,6)
      Fkm += cart.train
    }

    BTModel(Fkm)
  }

  def pk(x: Feature,
      Fkm: ArrayBuffer[CARTRegModel]): Double = {
    //Fkm: ArrayBuffer[DSModel]): Double = {
    //var fork = 0.0
//    (Fkm.map(model => {
//      model.getRegValue(x,
//        !insts.numIdx.contains(model.node.iAttr), model.node.iAttr, model.node.split)
//    }).sum+0.0)
    
        (Fkm.map(model => {
      model.getRegValue(x,insts.numIdx,1)
    }).sum+0.0)
  }
  case class BTModel(Fkm: ArrayBuffer[CARTRegModel]) extends Model {

    def predict(test: Instances): Double = {
      var r = 0.0
      val numIdx = test.numIdx
      test.data.map(lf => {
        val c = pk(lf, Fkm)
        println(lf.target + "=>" + c)
        r += (lf.target.toDouble - c) * (lf.target.toDouble - c)
      })
      r / test.data.size
    }
  }
}
object BoostingTree {

  def main(args: Array[String]): Unit = {

    var numIdx = new HashSet[Int]
    numIdx.+=(0)
    numIdx.+=(1)
    numIdx.+=(2)
    numIdx.+=(3)
    numIdx.+=(4)
    numIdx.+=(5)
//        numIdx.+=(7)
//        numIdx.+=(8)
//        numIdx.+=(10)

    val insts = new Instances(numIdx)
    insts.read("E:/books/spark/ml/decisionTree/cpu.csv")
    //    insts.all2Normal
    val (trainset, testset) = insts.stratify()

    val t = new BoostingTree(trainset, 21)
    val model = t.train()

    val loss = model.predict(testset)
    println(loss);
  }
}