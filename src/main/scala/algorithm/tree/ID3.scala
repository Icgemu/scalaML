package algorithm.tree

import scala.io.Source
import java.io.File
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashMap
import scala.collection.mutable.HashSet
import scala.Array.canBuildFrom
object ID3 {
  case class LabeledFeature(label: String, features: Array[String])
  case class Node(i: Int, sets: HashMap[String, Node], label: String)
  var idx = HashMap[Int,HashSet[String]]()
  var data = ArrayBuffer[LabeledFeature]()
  var root = Node(-1, null, "")
  var ite = 0
  def index(file: String) :Node = {
    var buff = Source.fromFile(new File(file))
    buff.getLines.toArray.map(line => {
      val arr = line.split(",")
      val label = arr.last.trim()
      val features = arr.slice(0, arr.length - 1)
      for (i <- 0 until features.length) {
        val v = features(i)
        if(!idx.contains(i)){idx.put(i,new HashSet[String]())}
        if (!idx(i).contains(v)) { idx(i)+=v.trim() }
      }
      data.+=(LabeledFeature(label, features))
    })
    buff.close
    buildDT(data,idx)
  }

  def buildDT(data: ArrayBuffer[LabeledFeature], idx: HashMap[Int,HashSet[String]]): Node = {
    println("ite:"+ ite)
    ite += 1
    var labels = new HashMap[String, Int]()
    data.map(f => {
      if (!labels.contains(f.label)) labels.+=((f.label, 1)) else labels(f.label)=labels(f.label) + 1
    })
    var sum = labels.values.sum
    //如果某改节点的某种分类占比大于0.95，则不继续分裂
    val filters = labels.map(l => (l._1, l._2 * 1.0 / sum)).filter(t => t._2 > 0.95).toArray
    if (filters.length > 0) {
      Node(-1, null, filters(0)._1)
    } else {
      //该节点总信息量
      val entropy = labels.map(f => { 
        val p =(f._2 *1.0 / sum)
       val r= -1* p* Math.log(p) 
       r
       }).sum
       //目前为止还未使用属性的各个属性信息量
      val f_entropy = new HashMap[Int,Double]

      for (i <- idx.keys) {
        val s = idx(i)
        var cnt = new HashMap[String, HashMap[String, Int]]()
        s.map(f => cnt(f) = new HashMap[String, Int])
        data.map(l => {
          val f = l.features(i)
          val label = l.label
          val n = cnt(f).getOrElse(label, 0) + 1
          cnt(f).put(label, n)
        })
        var entr = new HashMap[String, Double]()
        var entr_cnt = new HashMap[String, Int]()
        cnt.map(f => {
          val feature = f._1
          val feature_cnt = f._2.values.sum
          entr_cnt(feature) = f._2.values.sum + entr_cnt.getOrElse(feature, 0)
          entr(feature) = f._2.map(t => { 
            val p = t._2 *1.0 / feature_cnt
            -1.0* p * math.log(p) 
           }).sum
        })
        val total = entr_cnt.values.sum
        f_entropy(i) = cnt.map(f => {
          val sum1 = f._2.values.sum
          val r = (sum1*1.0 / total) * entr(f._1)
          r
        }).sum
      }

      //找出最大信息增益
      var finals = f_entropy.map(f =>{(f._1, entropy - f._2)})
      var maxval = (-1.0)
      var maxindex = (-1)
      for (i <- finals.keys) {
        if (finals(i) > maxval) { maxval = finals(i); maxindex = i }
      }

      //val splitFeature = idx(maxindex)

      idx.remove(maxindex)
      var sets = new HashMap[String, Node]()

      //对数据进行分裂，递归计算
      var splitData = new HashMap[String, ArrayBuffer[LabeledFeature]]
      data.map(f => {
        val ty = f.features(maxindex)
        //f.features.drop(n)
        if (!splitData.contains(ty)) { splitData.put(ty, new ArrayBuffer[LabeledFeature]) }
        splitData(ty).+=(f)
      })
      splitData.map(f => {
        sets(f._1) = buildDT(f._2, idx)
      })
      //    splitData.map(f=>{
      //      val ty = f._1
      //      val va = f._2
      //      var label_cnt = new HashMap[String,Int]
      //      va.map(l=>{
      //        label_cnt(l.label) = label_cnt.getOrElse(l.label, 0)+1
      //      })
      //      val s = label_cnt.values.sum
      //      val filter = label_cnt.map(l=>l._2/s).filter(t=>t>0.9).toArray
      //      (ty,filter.size>0)
      //    })
      //    .map(t=>if(t._2) {splitData.remove(t._1)})

      Node(maxindex, sets, "")
    }
  }

  def main(args: Array[String]): Unit = {
    
    val t = index("E:/books/spark/ml/decisionTree/test.txt")
    
    t
  }

}