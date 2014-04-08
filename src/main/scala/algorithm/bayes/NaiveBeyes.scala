package algorithm.bayes

import scala.collection.mutable.HashMap
import scala.collection.mutable.HashSet
import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import java.io.File
import scala.Array.canBuildFrom

object NaiveBeyes {
  case class LabeledFeature(label: String, features: Array[String])

  var yi_prob = new HashMap[String, Double]
  var ai_yi_prob = new HashMap[String, Array[HashMap[String, Double]]]

  var idx = HashMap[Int, HashSet[String]]()
  var data = ArrayBuffer[LabeledFeature]()
  var numIdx = new HashSet[Int]
  var labels = new HashSet[String]

  def index(file: String) = {
    var buff = Source.fromFile(new File(file))
    buff.getLines.toArray.map(line => {
      val arr = line.split(",")
      val label = arr.last.trim()
      labels.+=(label)
      val features = arr.slice(0, arr.length - 1)
      for (i <- 0 until features.length) {
        var v = features(i)
        //if("?".equalsIgnoreCase(v)){v="_NAN_"}
        if (!idx.contains(i)) { idx.put(i, new HashSet[String]()) }
        if (!idx(i).contains(v) && !numIdx.contains(i)) { idx(i) += v.trim() }
      }
      data.+=(LabeledFeature(label, features.map(f => f.trim())))
    })
    buff.close
    labels.map(lb => ai_yi_prob.put(lb, Array.fill(idx.size)(new HashMap[String, Double])))
    train()
  }

  def train() {
    data.map(lb => {
      val features = lb.features
      val label = lb.label
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
      yi_prob(label) = (sum_class + 1) / (class_all + yi_prob.size)
    })

    data.map(lb => {
      val features = lb.features
      val label = lb.label
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
    labels.map(label => {
      for (i <- numIdx.iterator) {
        val cnt = ai_yi_prob(label)(i)("count")
        val mean = ai_yi_prob(label)(i)("mean")
        val dev = ai_yi_prob(label)(i)("dev")
        ai_yi_prob(label)(i)("mean") = mean / cnt
        ai_yi_prob(label)(i)("dev") = math.sqrt(dev / (cnt - 1))
      }
    })
  }

  def classify(f: LabeledFeature) {
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
    println(f.label + "=>" + maxLabel)
  }
  def main(args: Array[String]): Unit = {
    //    numIdx.+=(0)
    //    numIdx.+=(1)
    //    numIdx.+=(2)
    //    numIdx.+=(3)
    //    numIdx.+=(5)
    //    numIdx.+=(7)
    //    numIdx.+=(8)
    //    numIdx.+=(10)

    val t = index("E:/books/spark/ml/decisionTree/bare.txt")

    data.map(d => classify(d))
  }

}
