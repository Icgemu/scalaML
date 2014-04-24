package classifier.bayes

import scala.collection.mutable.HashMap
import scala.collection.mutable.HashSet
import scala.collection.mutable.ArrayBuffer
import classifier.Classifier
import classifier.Model
import core.Instances

class NaiveBeyes(insts: Instances) extends Classifier {

  var yi_prob = new HashMap[String, Double]
  var ai_yi_prob = new HashMap[String, Array[HashMap[String, Double]]]

  var idx = insts.idxForNominal
  var data = insts.data
  var numIdx = insts.numIdx
  var labels = insts.labels
  labels.map(lb => ai_yi_prob.put(lb, Array.fill(idx.size)(new HashMap[String, Double])))

  def train(): Model = {

    data.map(lb => {
      val features = lb.features
      val label = lb.target
      yi_prob(label) = yi_prob.getOrElse(label, 0.0) + 1
      for (i <- 0 until features.length) {
        if (numIdx.contains(i)) {
          val mean = ai_yi_prob(label)(i).getOrElse("mean", 0.0)
          ai_yi_prob(label)(i)("mean") = mean + features(i).toDouble
          val cnt = ai_yi_prob(label)(i).getOrElse("count", 0.0)
          ai_yi_prob(label)(i)("count") = cnt + 1
        } else {
          val cnt = ai_yi_prob(label)(i).getOrElse(features(i), 0.0)
          ai_yi_prob(label)(i)(features(i)) = cnt + 1
        }
      }
    })

    //index for Nominal
    val noridx = idx.filter(p => p._2.size > 0)

    val class_all = yi_prob.values.sum
    labels.map(label => {
      val sum_class = yi_prob(label)
      for (i <- noridx.keys.iterator) {
        val nor = noridx(i)
        for (j <- nor.iterator) {
          val cnt = ai_yi_prob(label)(i).getOrElse(j, 0.0)
          ai_yi_prob(label)(i)(j) = (cnt + 1) / (sum_class + noridx(i).size)
        }
      }
      //
      yi_prob(label) = (sum_class + 1) / (class_all + yi_prob.size)
    })

    data.map(lb => {
      val features = lb.features
      val label = lb.target
      //yi_prob(label) = yi_prob.getOrElse(label, 0.0) + 1
      for (i <- numIdx.iterator) {
        var sum = ai_yi_prob(label)(i)("mean")
        var cnt = ai_yi_prob(label)(i)("count")
        val avg1 = sum / cnt
        //ai_yi_prob(label)(i)("mean") = avg1
        val dev = ai_yi_prob(label)(i).getOrElse("dev", 0.0) +
          (features(i).toDouble - avg1) * (features(i).toDouble - avg1)

        ai_yi_prob(label)(i)("dev") = dev
      }
    })
    //model for continuous attr  
    labels.map(label => {
      for (i <- numIdx.iterator) {
        val cnt = ai_yi_prob(label)(i)("count")
        val mean = ai_yi_prob(label)(i)("mean")
        val dev = ai_yi_prob(label)(i)("dev")
        ai_yi_prob(label)(i)("mean") = mean / cnt
        ai_yi_prob(label)(i)("dev") = math.sqrt(dev / (cnt - 1))
      }
    })

    NBModel()

  }

  case class NBModel() extends Model {

    def predict(test: Instances): Double = {
      var r = 0.0
      test.data.foreach(f => {

        var prob = new HashMap[String, Double]
        val noridx = idx.filter(p => p._2.size > 0)
        labels.map(label => {
          for (i <- noridx.keys) {
            val c = noridx(i).iterator
            for (j <- c) {
              val p = prob.getOrElse(label, yi_prob(label))
              prob.put(label, p * ai_yi_prob(label)(i)(j))
            }
          }
          for (i <- numIdx.iterator) {

            val mean = ai_yi_prob(label)(i)("mean")
            val dev = ai_yi_prob(label)(i)("dev")
            val p = prob.getOrElse(label, yi_prob(label))
            val v = f.features(i).toDouble
            val g = ((1 / (math.sqrt(2 * math.Pi) * dev)) * (
              math.exp(-1 * (v - mean) * (v - mean) / (2 * dev * dev))))
            prob.put(label, p * g)

          }
        })
        var maxLabel = ""
        var maxprob = -1.0
        prob.map(p => {
          if (p._2 > maxprob) { maxLabel = p._1; maxprob = p._2 }
        })
        if (maxLabel.equalsIgnoreCase(f.target)) r += 1.0
        println(f.target + "=>" + maxLabel)

      })
      r / test.data.size
    }
  }
}

object NaiveBeyes {

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

    val t = new NaiveBeyes(trainset)
    val model = t.train()

    val accu = model.predict(testset)
    println(accu);
  }

}
