package algorithm

import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashSet
import scala.io.Source
import java.io.File

class RegInstances(
  num: HashSet[Int]) {
  //var numIdx = numIdx
  val idx = HashMap[Int, HashSet[String]]()
  val data = ArrayBuffer[RegFeature]()
  val numIdx = num
  //val labels = new HashSet[String]

  //  this(num1:HashSet[String],idx1:HashMap[Int, HashSet[String]],labels1:HashSet[String]){
  //    this(num1)
  //    
  //  }
  def attr(): Int = idx.size
  def attr(i: Int): HashSet[String] = idx(i)

  def read(file: String) {
    var buff = Source.fromFile(new File(file))
    var j = 0
    buff.getLines.toArray.map(line => {
      val arr = line.split(",")
      val label = arr.last.trim().toDouble
      //labels.+=(label)
      val features = arr.slice(0, arr.length - 1)
      for (i <- 0 until features.length) {
        var v = features(i)
        //if("?".equalsIgnoreCase(v)){v="_NAN_"}
        if (!idx.contains(i)) { idx.put(i, new HashSet[String]()) }
        if (!idx(i).contains(v) && !numIdx.contains(i)) { idx(i) += v.trim() }
      }
      data.+=(new RegFeature(j, label, features.map(f => f.trim()), 1.0))
      j += 1
    })
    buff.close
  }

  def read(x: Array[Array[String]], y: Array[Double]) {
    //var buff = Source.fromFile(new File(file))
    var j = 0
    x.map(line => {
      val arr = line
      val label = y(j)
      //labels.+=(label+"")
      val features = arr.slice(0, arr.length)
      for (i <- 0 until features.length) {
        var v = features(i)
        //if("?".equalsIgnoreCase(v)){v="_NAN_"}
        if (!idx.contains(i)) { idx.put(i, new HashSet[String]()) }
        if (!idx(i).contains(v + "") && !numIdx.contains(i)) { idx(i) += (v + "") }
      }
      data.+=(new RegFeature(j, label, features.map(f => f + ""), 1.0))
      j += 1
    })
    // buff.close
  }

  def addInstances(inst: RegFeature) {
    data.+=:(inst)
  }
  def intWeight() {
    //var weightSum = 0.0
    //data.map(f=>{weightSum += f.weight})
    data.map(f => { f.weight = f.value })
  }

  def sample(factor: Double): RegInstances = {
    val size = (data.size * factor).toInt
    val insts = new RegInstances(numIdx)
    for (i <- 0 to size) {
      val v = (data.size * math.random).toInt
      insts.addInstances(data.remove(v))
    }
    insts
  }
  def copy(): RegInstances = {
    val insts = new RegInstances(numIdx)
    for (inst <- data.toIterator) {
      // val v = (data.size * math.random).toInt
      insts.addInstances(inst)
    }
    insts
  }
  //  def reWeight(m:(Int,Double,String,String,String,Double)){
  //    val alpha = math.exp(m._2)
  //    if(numIdx.contains(m._1)){
  //      val split = m._3.toDouble
  //      data.map(d=>{
  //        if(!d.features(m._1).equalsIgnoreCase("?")){
  //        val f = d.features(m._1).toDouble
  //        val label = if(f>split){m._5}else{m._4}
  //        if(!label.equalsIgnoreCase(d.label)){d.weight = d.weight*alpha}
  //        }
  //      })
  //    }else{
  //      val split = m._3
  //      data.map(d=>{
  //        val f = d.features(m._1)
  //        val label = if(f.equalsIgnoreCase(split)){m._4}else{m._5}
  //        if(!label.equalsIgnoreCase(d.label)){d.weight = d.weight*alpha}
  //      })
  //    }
  //  }
}