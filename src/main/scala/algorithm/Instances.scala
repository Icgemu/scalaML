package algorithm

import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashSet
import scala.io.Source
import java.io.File

class Instances(
  num: HashSet[Int]) {
  //var numIdx = numIdx
  var idx = HashMap[Int, HashSet[String]]()
  val data = ArrayBuffer[LabeledFeature]()
  val numIdx = num
  val labels = new HashSet[String]

  //  this(num1:HashSet[String],idx1:HashMap[Int, HashSet[String]],labels1:HashSet[String]){
  //    this(num1)
  //    
  //  }
  def attr(): Int = idx.size
  def attr(i: Int): HashSet[String] = idx(i)

  def numClass(): Int = labels.size
  def classof(): HashSet[String] = labels

  def classToInt(lf: LabeledFeature): Int = {
    val label = lf.label
    var i = 0
    var j = 0
    labels.map(f => { if (f.equalsIgnoreCase(label)) { j = i }; i += 1 })
    j
  }
  def classToInt(lf: String): Int = {
    val label = lf
    var i = 0
    var j = 0
    labels.map(f => { if (f.equalsIgnoreCase(label)) { j = i }; i += 1 })
    j
  }
  def read(file: String) {
    var buff = Source.fromFile(new File(file))
    var j = 0
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
      data.+=(new LabeledFeature(j, label, features.map(f => f.trim()), 1.0))
      j += 1
    })
    buff.close
  }

  def read(x: Array[Array[Double]], y: Array[Int]) {
    //var buff = Source.fromFile(new File(file))
    var j = 0
    x.map(line => {
      val arr = line
      val label = y(j)
      labels.+=(label + "")
      val features = arr.slice(0, arr.length)
      for (i <- 0 until features.length) {
        var v = features(i)
        //if("?".equalsIgnoreCase(v)){v="_NAN_"}
        if (!idx.contains(i)) { idx.put(i, new HashSet[String]()) }
        if (!idx(i).contains(v + "") && !numIdx.contains(i)) { idx(i) += (v + "") }
      }
      data.+=(new LabeledFeature(j, label + "", features.map(f => f + ""), 1.0))
      j += 1
    })
    // buff.close
  }
  def read(x: Array[Array[Double]], y: Array[String]) {
    //var buff = Source.fromFile(new File(file))
    var j = 0
    x.map(line => {
      val arr = line
      val label = y(j)
      labels.+=(label + "")
      val features = arr.slice(0, arr.length)
      for (i <- 0 until features.length) {
        var v = features(i)
        //if("?".equalsIgnoreCase(v)){v="_NAN_"}
        if (!idx.contains(i)) { idx.put(i, new HashSet[String]()) }
        if (!idx(i).contains(v + "") && !numIdx.contains(i)) { idx(i) += (v + "") }
      }
      data.+=(new LabeledFeature(j, label, features.map(f => f + ""), 1.0))
      j += 1
    })
    // buff.close
  }
  def read(x: Array[Array[String]], y: Array[String]) {
    //var buff = Source.fromFile(new File(file))
    var j = 0
    x.map(line => {
      val arr = line
      val label = y(j)
      labels.+=(label + "")
      val features = arr.slice(0, arr.length)
      for (i <- 0 until features.length) {
        var v = features(i)
        //if("?".equalsIgnoreCase(v)){v="_NAN_"}
        if (!idx.contains(i)) { idx.put(i, new HashSet[String]()) }
        if (!idx(i).contains(v + "") && !numIdx.contains(i)) { idx(i) += (v + "") }
      }
      data.+=(new LabeledFeature(j, label, features.map(f => f + ""), 1.0))
      j += 1
    })
    // buff.close
  }
  def addInstances(inst: LabeledFeature) {
    data.+=:(inst)
  }
  def intWeight() {
    var weightSum = 0.0
    data.map(f => { weightSum += f.weight })
    data.map(f => { f.weight = f.weight / weightSum })
  }

  def sample(factor: Double): Instances = {
    val size = (data.size * factor).toInt
    val insts = new Instances(numIdx)
    for (i <- 0 to size) {
      val v = (data.size * math.random).toInt
      insts.addInstances(data.remove(v))
    }
    insts
  }
  def copy(): Instances = {
    val insts = new Instances(numIdx)
    for (inst <- data.toIterator) {
      // val v = (data.size * math.random).toInt
      insts.addInstances(inst)
    }
    insts
  }
  def reWeight(m: (Int, Double, String, String, String, Double)) {
    val alpha = math.exp(m._2)
    if (numIdx.contains(m._1)) {
      val split = m._3.toDouble
      data.map(d => {
        if (!d.features(m._1).equalsIgnoreCase("?")) {
          val f = d.features(m._1).toDouble
          val label = if (f > split) { m._5 } else { m._4 }
          if (!label.equalsIgnoreCase(d.label)) { d.weight = d.weight * alpha }
        }
      })
    } else {
      val split = m._3
      data.map(d => {
        val f = d.features(m._1)
        val label = if (f.equalsIgnoreCase(split)) { m._4 } else { m._5 }
        if (!label.equalsIgnoreCase(d.label)) { d.weight = d.weight * alpha }
      })
    }
  }
}