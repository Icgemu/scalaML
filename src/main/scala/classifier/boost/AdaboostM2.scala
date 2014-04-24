package classifier.boost

import classifier.Classifier
import classifier.Model
import core.Instances
import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuffer
import classifier.tree.DSModel
import classifier.tree.DicisionStump
import scala.collection.mutable.HashSet

//http://www.public.asu.edu/~jye02/CLASSES/Fall-2005/PAPERS/boosting-icml.pdf
//experiments with a new boosting algorithm: Adaboost M2 algorithm
class AdaboostM2(insts: Instances, M: Int) extends Classifier {

  def train(): Model = {

    val K = insts.classof.size
    val Ks = insts.classof.toArray
    val data = insts.data
    val size = data.size

    //weight for y != yi
    val Wty = Array.fill(data.size)(new HashMap[String, Double]())
    var i = 0
    data.map(f => {
      val label = f.target
      val w = f.weight
      Ks.map(cl => {
        if (!cl.equalsIgnoreCase(label)) { Wty(i)(cl) = w / ((K - 1)) }
      })
      i += 1
    })

    //Models
    val Gm = ArrayBuffer[(Double, DSModel)]()
    for (m <- 0 until M) {
      //normalize on Wty where Y != Yi
      val Qty = Array.fill(size)(new HashMap[String, Double]())
      //sum on Wty for each sample
      val Wt = Array.fill(size)(0.0)
      for (i <- 0 until size) {
        Wt(i) = Wty(i).values.sum
        Qty(i) = Wty(i).map(f => (f._1, f._2 / Wt(i)))
      }

      //all Training sets summarization
      val sum = Wt.sum
      //weight for each sample in this iteration
      val Dt = Array.fill(size)(0.0)
      for (i <- 0 until size) {
        Dt(i) = Wt(i) / sum
        data(i).weight = Dt(i)
      }

      // train on current weight
      val t = new DicisionStump(insts)
      val model = t.train()
      var err = 0.0
      
      //calculating err 
      for (i <- 0 until data.size) {
        //get hypothesis ht(Xi)
        val m = model.getLabelDist(data(i), 
            !insts.numIdx.contains(model.node.iAttr), model.node.iAttr, model.node.split)
        var ht = 0.0
        var hty = 0.0

        val t = m.map(f => {
          val t = f._2
          if (f._1.equalsIgnoreCase(data(i).target)) {
            ht = t
          } else {
            hty += t * Qty(i).getOrElse(f._1, 0.0)
          }
        })
        err += 0.5 * Dt(i) * (1 - ht + hty)
      }
      
      // small errs ->small beta ->very big (math.log(1.0 / beta)
      if (err > 1e-4) {
        val beta = err / (1 - err)
        print(err + "," + beta + ",")
        for (i <- 0 until data.size) {
          val m = model.getLabelDist(data(i), 
              !insts.numIdx.contains(model.node.iAttr), model.node.iAttr, model.node.split)

          //update weight
          Wty(i).map(f => {
            val t = m.getOrElse(f._1, 0.0)
            Wty(i)(f._1) = Wty(i).getOrElse(f._1, 0.0) *
              Math.pow(beta, 0.5 * (1 + m.getOrElse(data(i).target, 0.0) - t))
          })
          
        }
        println(math.log(1.0 / beta))
        //add model for this iteration
        Gm += ((math.log(1.0 / beta), model))
      }
    }
    M2Model(Gm)
  }

  case class M2Model(models: ArrayBuffer[(Double, DSModel)]) extends Model {

    def predict(test: Instances): Double = {

      var r = 0.0
      val numIdx = test.numIdx
      val Ks = test.classof.toArray
      test.data.map(lf => {
        //distribution of Xi over all models
        var dist = HashMap[String, Double]()
        
        models.map(bt => {
          val alpha = bt._1
          val model = bt._2
          val m = model.getLabelDist(lf, !
              test.numIdx.contains(model.node.iAttr), model.node.iAttr, model.node.split)
          m.map(f => {
            //ht(Xi)
            val t = f._2
            
            dist(f._1) = dist.getOrElse(f._1, 0.0) + alpha * t
          })
        })

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

object AdaboostM2 {

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

    val t = new AdaboostM2(trainset, 21)
    val model = t.train()

    val accu = model.predict(testset)
    println(accu);
  }
}