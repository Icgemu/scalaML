package algorithm.boost

import algorithm.Instances
import algorithm.LabeledFeature
import scala.collection.mutable.HashMap
import algorithm.RegInstances
import algorithm.tree.CARTReg
import scala.collection.mutable.ArrayBuffer
import algorithm.tree.DecisionStump.Node
import scala.collection.mutable.HashSet
import algorithm.RegFeature
import algorithm.tree.DecisionStump

//Gradient Boosting Tree
object GradientBoost {

  def classifier(insts: Instances, M: Int, J: Int) = {
    val K = insts.classof.size
    val Ks = insts.classof.toArray

    //insts.intWeight
    var Fkm = ArrayBuffer[HashMap[String, HashMap[Int, Node]]]()
    var Fk0 = HashMap[String, Double]()
    Ks.map(f => Fk0(f) = math.random)
    var is = true
    var y = Double.MaxValue
    //while(is){
    for (i <- 0 until M) {
      //sum = y
      println("MM----------------MM")
      val tmp = HashMap[String, HashMap[Int, Node]]()
      for (iclass <- Ks.iterator) {
        //val k = insts.classToInt(iclass)
        val X = Array.fill(insts.data.size)(Array.fill(insts.attr)(""))
        val Y = Array.fill(insts.data.size)(0.0)
        var sum = 0.0
        for (i <- 0 until insts.data.size) {
          val inst = insts.data(i)

          X(i) = inst.features
          val c = pk(inst, Ks, iclass, Fk0, Fkm, insts.numIdx)
          //println(c)
          //if(i==0){println(inst.features.mkString(",")+"=>"+c)}
          val yik = (if (inst.label.equalsIgnoreCase(iclass)) 1.0 else 0.0) - c
          //println(yik)
          sum += yik * yik
          //inst.weight = yik
          //注意是梯度的负方向
          //Y(i) = -1*yik
          // println(yik)
          Y(i) = yik
        }
        println("Log=" + sum)
        //      if(sum < 0.1){
        //        is = false
        //      }else{
        //        y = sum
        //      }
        val reg = new RegInstances(insts.numIdx)
        reg.read(X, Y)
        //reg.data.map(f=>println(f.features.mkString(",")+","+f.value))
        val nodes = DecisionStump.classifier(reg, J)
        DecisionStump.printTree(nodes, nodes(1), 0)
        tmp(iclass) = nodes
      }
      Fkm += tmp
    }

    test(insts, Fk0, Fkm)
  }

  def pk(x: LabeledFeature, Ks: Array[String], iclass: String,

    Fk0: HashMap[String, Double],
    Fkm: ArrayBuffer[HashMap[String, HashMap[Int, Node]]], numIdx: HashSet[Int]): Double = {
    var fork = 0.0
    var sum = 0.0
    for (i <- Ks.toIterator) {
      //println("---")
      val t = (Fkm.map(f => {
        val nodes = f(i)
        //CARTReg.printTree(nodes(1), 0)
        val c = DecisionStump.instanceFor(nodes, 1, x.features, numIdx)
        ///println(x.features.mkString(",")+"==>"+c)
        val r = ((Ks.size - 1) * 1.0 / Ks.size) * c
        //if(i==0)println(c)
        //val r = c
        r
      }).sum + Fk0(i))
      //println(t)
      sum += math.exp(t)
      //println(t +"=>" +sum)
      if (i.equalsIgnoreCase(iclass)) {
        fork = math.exp(t)
      }
    }
    //println(fork+"/"+sum)
    //    if(sum<1e-10){
    //      0.0
    //    }else{
    fork / sum
    //    }
    //fork

    //t
  }

  def test(insts: Instances,
    Fk0: HashMap[String, Double],
    Fkm: ArrayBuffer[HashMap[String, HashMap[Int, Node]]]) {
    val numIdx = insts.numIdx
    val Ks = insts.classof.toArray
    insts.data.map(lf => {
      val p = HashMap[String, Double]()
      for (k <- Ks.toIterator) {
        p(k) = pk(lf, Ks, k, Fk0, Fkm, numIdx)
        //val c = pk(lf,-1,Fk0,Fkm,numIdx)
      }
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

    classifier(insts, 20, 2)
  }

}