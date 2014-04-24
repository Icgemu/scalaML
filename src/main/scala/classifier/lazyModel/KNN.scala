package classifier.lazyModel

import classifier.Classifier
import classifier.Model
import core.Instances
import core.Feature
import scala.collection.mutable.HashSet

class KNN(insts: Instances, K: Int) extends Classifier {

  def train(): Model = {
    KNNModel()
  }

  case class KNNModel() extends Model {
    def distance(a1: Feature, a2: Feature): Double = {

      val f1 = a1.features
      val f2 = a2.features
      val d = f1.zip(f2).map(t => {
        val t1 = t._1.toDouble
        val t2 = t._2.toDouble
        (t1 - t2) * (t1 - t2)
      }).sum
      math.sqrt(d)
    }

    def predict(test: Instances): Double = {
      var r = 0.0
      for (i <- 0 until test.data.size) {
        var topK = new HashSet[(Double, Int)]
        for (j <- 0 until insts.data.size) {
          topK.add((distance(test.data(i), insts.data(j)), j))
        }
        val k = topK.toArray
        val sort = k.sortBy(f => f._1).slice(0, K)
        val labels = sort.map(f => { insts.data(f._2) })
        val dist = labels.groupBy(f => { f.target }).map(f => { (f._1, f._2.size) })
        val rst = dist.toArray.sortBy(f=>f._2).reverse
        val label = rst(0)._1
        if(label.equalsIgnoreCase(test.data(i).target)) r+=1.0
        println(test.data(i).target + "=>" + dist)
      }
      r/test.data.size
    }
  }
}
object KNN {

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
    //insts.all2Normal
    val (trainset, testset) = insts.stratify()

    val t = new KNN(trainset,3)
    val model = t.train()

    val accu = model.predict(testset)
    println(accu);
  }
}