package algorithm.ann

import algorithm.Instances
import algorithm.cluster.SimpleKMeans
import algorithm.LabeledFeature
import algorithm.regression.RidgeRegression
import scala.collection.mutable.HashSet

object RBFNetwork {
  
  def classifier(inst:Instances,K:Int, T: Int, rate: Double, fold: Int){
    
    val m = SimpleKMeans.kmean(inst.data, K)
    val centers = m.centers
    val k = m.K
    val p = m.par
    
    val std = p.map(f=>{
      val index = f._1
      val data = f._2
      val std1 = data.map(lf=>{
        val arr = lf.features.map(xi=>{xi.toDouble})
        val s = arr.zip(centers(index)).map(t=>{
          (t._1 - t._2)*(t._1 - t._2)
        }).sum
        s
      }).sum
      std1/(data.size -1)
    }).toArray
    
    val data = inst.data
    
    val x = data.map(f=>{
      val t = Array.fill(centers.size)(0.0)
      for(i<- 0 until centers.size){
        t(i) = guass(f,centers(i),std(i))
      }
      (t)
    }).toArray
    val y = data.map(f=>f.label).toArray
    val insts = new Instances(inst.numIdx)
    insts.read(x, y)
    
    RidgeRegression.classifier(insts, T, rate, fold)
  }
  
  def guass(xi:LabeledFeature,c:Array[Double],std:Double):Double ={
    val x = xi.features.map(f=>f.toDouble)
    val sum = x.zip(c).map(f=>{
      (f._1 - f._2)*(f._1 - f._2)
    }).sum
    math.exp(-1* sum / (2 * std))
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

    classifier(insts,9, 100, 0.5, 10)
  }

}