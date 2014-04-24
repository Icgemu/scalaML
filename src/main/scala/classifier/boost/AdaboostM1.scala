package classifier.boost

import classifier.Classifier
import classifier.Model
import core.Instances
import classifier.tree.DicisionStump
import classifier.tree.DSModel
import scala.collection.mutable.HashMap
import scala.collection.mutable.HashSet


//http://www.public.asu.edu/~jye02/CLASSES/Fall-2005/PAPERS/boosting-icml.pdf
//experiments with a new boosting algorithm : Adaboost M1 algorithm
class AdaboostM1(insts: Instances, M: Int) extends Classifier {

  def train(): Model = {

    //insts.intWeight()
    val Gm: Array[(DSModel, Double)] = Array.fill(M)(null)
    var isreturn = false
    var i = 0
    while (i < M && !isreturn) {
      val tol = insts.data.map(f => f.weight).sum
      insts.data.foreach(f => f.weight = f.weight / tol)

      val t = new DicisionStump(insts)
      val model = t.train()
      val err = 1.0 - model.predict(insts)
      //val err = m._6
      if (err >= 0.5) {
        println(i + ": err =" + err)
        isreturn = true
      } else {
        
        val beta = err/(1.0-err)
        //val alpha = math.log((1 - err) / err)
        val alpha = math.log(1.0/ beta)
        Gm(i) = (model, alpha)
        val iAttr = model.node.iAttr
        val split = model.node.split

        println(err+","+beta+","+alpha)
        insts.data.map(d => {
          val label = model.getLabelValue(d, !insts.numIdx.contains(iAttr), iAttr, split)
          if (label.equalsIgnoreCase(d.target)) { d.weight = d.weight * beta }
        })

      }
      i += 1
    }
    val Gmf = Gm.slice(0, if(isreturn)i-1 else i)
    M1Model(Gmf)
  }

  case class M1Model(Gm: Array[(DSModel, Double)]) extends Model {

    def predict(test: Instances): Double = {

      var r = 0.0
      val data = insts.data
      test.data.map(f => {
        val l = f.target

        val dist = HashMap[String, Double]()
       
        Gm.map(gm => {

          val iAttr = gm._1.node.iAttr
          val alpha = gm._2
          val splitFor = gm._1.node.split

          val label = gm._1.getLabelValue(f, !test.numIdx.contains(iAttr), iAttr, splitFor)
          dist(label) = dist.getOrElse(label, 0.0) + alpha
        })
        
        val sum = dist.values.sum
        val nor = dist.map(f => {
          (f._1, f._2 / sum)
        })
        val rst = nor.toArray.sortBy(f=>f._2).reverse
        val label = rst(0)._1
        if(label.equalsIgnoreCase(l)) r+=1.0
        println(l + "=>" + nor)        
      })

      r / test.data.size
    }
  }
}
object AdaboostM1 {
  

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

    val t = new AdaboostM1(trainset,21)
    val model = t.train()

    val accu = model.predict(testset)
    println(accu);
  }
}