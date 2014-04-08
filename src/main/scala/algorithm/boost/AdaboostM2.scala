package algorithm.boost

import algorithm.Instances
import scala.collection.mutable.HashMap
import algorithm.tree.ClassificationDecisionStump2
import scala.collection.mutable.ArrayBuffer
import algorithm.tree.ClassificationDecisionStump2.Node
import scala.collection.mutable.HashSet

object AdaboostM2 {

  def classifier(insts: Instances, M: Int) = {

    val K = insts.classof.size
    val Ks = insts.classof.toArray
    val data = insts.data
    val size = data.size

    val Wty = Array.fill(data.size)(new HashMap[String, Double]())
    data.map(f => {
      val label = f.label
      val w = f.weight
      val i = f.i
      Ks.map(cl => {
        //if (!cl.equalsIgnoreCase(label)) { Wty(i)(cl) = w / (size * (K - 1)) }
        if (!cl.equalsIgnoreCase(label)) { Wty(i)(cl) = w / ( (K - 1)) }
      })
    })

    val Dt = Array.fill(size)(0.0)
    val Gm = ArrayBuffer[(Double, HashMap[Int, Node])]()
    for (m <- 0 until M) {
      println("MM----------------MM")
      val Qty = Array.fill(size)(new HashMap[String, Double]())
      val Wt = Array.fill(size)(0.0)
      for (i <- 0 until size) {
        val lf = data(i)
        Wt(i) = Wty(i).values.sum
        Qty(i) = Wty(i).map(f => (f._1, f._2 / Wt(i)))
      }
      
      val sum = Wt.sum
      for (i <- 0 until size) {
        //print(Dt(i)+"=>")
        Dt(i) = Wt(i) / sum
       // println(Dt(i))
        data(i).weight = Dt(i)
      }

      val nodes = ClassificationDecisionStump2.classifier(insts, 2)
      ClassificationDecisionStump2.printTree(nodes, nodes(1), 0)
      var err = 0.0
      for (i <- 0 until data.size) {
        val m = ClassificationDecisionStump2.instanceFor(nodes, 1, data(i).features, insts.numIdx)
        val sum = m.values.sum
        var ht = 0.0
        var hty = 0.0

        val t = m.map(f => {
          val t = f._2 / sum
          //val t = f._2 
          if (f._1.equalsIgnoreCase(data(i).label)) {
            ht = t
          } else {
            hty += t * Qty(i).getOrElse(f._1, 0.0)
          }
        })
        err += 0.5 * Dt(i) * (1 - ht + hty)
      }
      val beta = err / (1 - err)
      //println(beta)
      for (i <- 0 until data.size) {
        val m = ClassificationDecisionStump2.instanceFor(nodes, 1, data(i).features, insts.numIdx)
        val sum = m.values.sum
       
        m.map(f => {
          val t = f._2 / sum
          //val t = f._2 
          if (!f._1.equalsIgnoreCase(data(i).label)) {
            //print(Wty(i).getOrElse(f._1,0.0) +"=>")
            Wty(i)(f._1) = Wty(i).getOrElse(f._1, 0.0) *
            //Math.pow(beta, 0.5 * (1 + m.getOrElse(data(i).label, 0.0) - t))
              Math.pow(beta, 0.5 * (1 + m.getOrElse(data(i).label, 0.0) / sum - t))
            //println(Wty(i)(f._1))
          }
        })
      }
      //println(math.log(1.0 / beta))
      Gm += ((math.log(1.0 / beta), nodes))
    }
    test(insts, Gm)
  }

  def test(insts: Instances,
    Fkm: ArrayBuffer[(Double, HashMap[Int, Node])]) {
    val numIdx = insts.numIdx
    val Ks = insts.classof.toArray
    insts.data.map(lf => {
      var p = HashMap[String, Double]()
      //for(k<- Ks.toIterator){
      //p(k) += pk(lf,Ks,k,Fk0,Fkm,numIdx)
      Fkm.map(bt => {
        val b = bt._1
        val f = bt._2
        val m = ClassificationDecisionStump2.instanceFor(f, 1, lf.features, numIdx)
        val sum = m.values.sum
        m.map(f => {
          //val t = f._2 / sum
          val t = f._2 
          p(f._1) = p.getOrElse(f._1, 0.0) + b * t
        })
      })
      //}

      //       }
      val sum = p.values.sum
      p = p.map(f => { (f._1, f._2 / sum) })
      for (k <- Ks.toIterator) {
        print(k + "=" + p.getOrElse(k, 0.0) + ",")
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

    classifier(insts, 30)
  }

}