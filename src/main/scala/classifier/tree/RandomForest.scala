package classifier.tree

import classifier.Model
import core.Instances
import scala.collection.mutable.HashMap
import classifier.tree.CARTModel
import scala.collection.mutable.HashSet

class RandomForest(insts:Instances,T:Int,numAttr:Int) extends TreeClassifierBase {
  
  def train():Model={
    
    val trees = Array.fill(T)(CARTModel(null))
    for (t <- 0 until T) {
      val N = insts.data.size

      val X = Array.fill(N)(Array.fill(insts.attr)(""))
      val Y = Array.fill(N)("")
      for (n <- 0 until N) {
        val i = (math.random * N).toInt
        X(n) = insts.data(i).features
        Y(n) = insts.data(i).target
      }

      val r = insts.attr - numAttr
      val idx = insts.idxForNominal.map(f => f)
      val num = insts.numIdx.map(f=>f)
      while (idx.size > numAttr) {
        val keys = idx.keys.toArray
        val t = (keys.size * math.random).toInt
        idx.remove(keys(t))
        num.remove(keys(t))
      }

      val data = new Instances(num)
      data.read(X, Y)
      data.idxForNominal = idx

      val model = new CART(data,false)
      
      //RFtree.printTree(nodes, nodes(1), 0)
      trees(t) = model.train
    }
    //test(insts, trees)
    RFModel(trees)
  }
  
  case class RFModel(models:Array[CARTModel]) extends Model{
    
    def predict(test: Instances): Double = {

      var r = 0.0
      test.data.map(lf => {              
        val label = models.map(m=>m.labelfor(lf, test))
        val sorted = label.groupBy(f=>f).map(f=>(f._1,f._2.size)).toArray.sortBy(f=>f._2).reverse
        println(sorted.mkString(",")+"=>"+lf.target)
        if ((sorted(0)._1).equalsIgnoreCase(lf.target)) r += 1.0

      })
      r / test.data.size

    }
  }
}

object RandomForest {
  

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
    //insts.all2Normal
    val (trainset, testset) = insts.stratify()

    val t = new RandomForest(trainset,20,10)
    val model = t.train()

    val accu = model.predict(testset)
    println(accu);
  }
}