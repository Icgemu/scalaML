package algorithm.model

import scala.collection.mutable.HashSet
import scala.Array.canBuildFrom
import algorithm.Instances
import algorithm.LabeledFeature

object KNN {

  def knn(instances: Instances, K: Int) {
    val test = instances.sample(0.2).data
    val train = instances.data
    for (i <- 0 until test.size) {
      var topK = new HashSet[(Double, Int)]
      for (j <- 0 until train.size) {
        topK.add((distance(test(i), train(j)), j))
      }
      val k = topK.toArray
      val sort = k.sortBy(f => f._1).slice(0, K)
      val labels = sort.map(f => { train(f._2) })
      val r = labels.groupBy(f => { f.label }).map(f => { (f._1, f._2.size) })

      println(test(i).label + "=>" + r)
    }
  }
  def distance(a1: LabeledFeature, a2: LabeledFeature): Double = {

    val f1 = a1.features
    val f2 = a2.features
    val d = f1.zip(f2).map(t => {
      val t1 = t._1.toDouble
      val t2 = t._2.toDouble
      (t1 - t2) * (t1 - t2)
    }).sum
    math.sqrt(d)
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
    knn(insts, 5)
  }

}