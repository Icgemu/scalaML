package algorithm.tree

import scala.io.Source
import java.io.File
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashMap
import scala.collection.mutable.HashSet
import scala.collection.immutable.TreeSet
import scala.Array.canBuildFrom
import scala.Option.option2Iterable
import algorithm.Instances
import algorithm.LabeledFeature
object C45v2 {
  //case class LabeledFeature(label: String, features: Array[String])
  case class Node(i: Int, split: Double, sets: HashMap[String, Node], label: String, hit: Int, mis: Int)
  //var idx = HashMap[Int, HashSet[String]]()
  //var data = ArrayBuffer[LabeledFeature]()
  // var numIdx = new HashSet[Int]
  //var root = Node(-1, null, "",0,0)
  //var ite = 0

  def classifier(insts: Instances): Node = {
    val data = insts.data
    val idx = insts.idx
    val numIdx = insts.numIdx
    buildDT(data, idx, numIdx)

    //prune()
  }
  def buildDT(data: ArrayBuffer[LabeledFeature],
    idx: HashMap[Int, HashSet[String]],
    numIdx: HashSet[Int]): Node = {
    //    println("ite:" + ite)
    //    ite += 1
    var labels = new HashMap[String, Int]()
    data.map(f => {
      if (!labels.contains(f.label)) labels.+=((f.label, 1)) else labels(f.label) = labels(f.label) + 1
    })
    var sum = labels.values.sum
    //如果某改节点的某种分类占比大于0.95，则不继续分裂
    val filters = labels.map(l => (l._1, l._2 * 1.0 / sum)).filter(t => t._2 > 0.95).toArray
    var split_point = new HashMap[Int, Double]
    if (filters.length > 0) {
      val hit = labels(filters(0)._1)
      val mis = labels.values.sum - hit
      Node(-1, -1.0, null, filters(0)._1, hit, mis)
    } else {
      //该节点总信息量
      val entropy = labels.map(f => {
        val p = (f._2 * 1.0 / sum)
        val r = -1 * p * log2(p)
        r
      }).sum
      //目前为止还未使用属性的各个属性信息量
      val f_entropy = new HashMap[Int, Double]
      val split_entropy = new HashMap[Int, Double]
      val dis_entropy = new HashMap[Int, Double]
      var remove_1 = new HashSet[Int]

      for (i <- idx.keys) {
        val s = idx(i)

        var cnt = new HashMap[String, HashMap[String, Double]]()
        var split_cnt = new HashMap[String, Double]()
        if (s.size > 0) { //标称属性
          s.map(f => cnt(f) = new HashMap[String, Double])
          data.map(l => {
            var f = l.features(i)
            //if("".equalsIgnoreCase(f)){f = "_NAN_"}
            val label = l.label
            val n = cnt(f).getOrElse(label, 0.0) + 1.0
            cnt(f).put(label, n)
            val n1 = split_cnt.getOrElse(f, 0.0) + 1.0
            split_cnt(f) = n1
          })
          var miss_cnt = cnt.remove("?") match {
            case None => 0
            case Some(t) => t.values.sum
          }
          split_cnt.remove("?")
          dis_entropy(i) = ((data.size - miss_cnt) * 1.0 / data.size)
          //处理缺失值问题
          //缺失的独立一个分类，对改属性的最终增益进行打折
          //          val miss_cnt = cnt.remove("_NAN_").get
          //          val miss_split_cnt = split_cnt.remove("_NAN_").get
          //          val a_cnt = split_cnt.values.sum
          //          cnt.map(f=>{
          //            val fraction = split_cnt(f._1)/a_cnt
          //            val ft = f._1
          //            miss_cnt.map(t=>{
          //              val label = t._1
          //              val c = t._2
          //              cnt(ft).put(label,cnt(ft)(label) + fraction* c)
          //              split_cnt.put(ft,split_cnt(ft) + fraction* miss_split_cnt)
          //            })
          //          })

          var entr = new HashMap[String, Double]()
          var entr_cnt = new HashMap[String, Double]()
          cnt.map(f => {
            val feature = f._1
            val feature_cnt = f._2.values.sum
            entr_cnt(feature) = f._2.values.sum + entr_cnt.getOrElse(feature, 0.0)
            entr(feature) = f._2.map(t => {
              val p = t._2 * 1.0 / feature_cnt
              -1.0 * p * log2(p)
            }).sum
          })
          var total = entr_cnt.values.sum
          //entr_cnt.remove("?")
          f_entropy(i) = cnt.map(f => {
            val sum1 = f._2.values.sum
            val r = (sum1 * 1.0 / total) * entr(f._1)
            r
          }).sum

          total = split_cnt.values.sum
          //split_cnt.remove("?")
          split_entropy(i) = split_cnt.map(f => {
            val p = f._2 * 1.0 / total
            -1.0 * p * log2(p)
          }).sum
        } else { //连续属性

          var sorted = new TreeSet[Double]()
          var nosorted = new TreeSet[Double]()
          var lastLabel = data(0).label
          var lastV = -1.0
          val sortmap = data.sortBy(f => { f.features(i) })

          sortmap.map(lf => {
            val v = lf.features(i)

            if (!lf.label.equals(lastLabel) && !"?".equalsIgnoreCase(v)) {

              sorted.+=(v.toDouble)
              //if()
              //sorted.+=(lastV)//只在class变化位置才会增加信息量

            }
            if (!"?".equalsIgnoreCase(v)) {
              lastLabel = lf.label
              lastV = v.toDouble
              nosorted += (lastV)
            }
          })
          //寻找信息值最大的分裂点
          if (sorted.size > 1) {
            val p = discrete(data, i, nosorted)
            f_entropy(i) = p._2
            split_entropy(i) = p._3
            split_point(i) = p._1
            dis_entropy(i) = p._4
          } else {
            remove_1.+=(i)
          }
        }
      }

      //找出最大信息增益率
      //找出增益均值以上的才进行计算增益率，避免数据偏移比较严重的分裂值很小
      val avg = f_entropy.map(f => { (entropy - f._2) * dis_entropy(f._1) }).sum / f_entropy.size
      var finals = f_entropy.filter(p => { (entropy - p._2) * dis_entropy(p._1) >= avg }).map(f => { (f._1, (entropy - f._2) * dis_entropy(f._1) / split_entropy(f._1)) })
      var maxval = (-1.0)
      var maxindex = (-1)
      for (i <- finals.keys) {
        if (finals(i) > maxval) { maxval = finals(i); maxindex = i }
      }

      remove_1.map(f => idx.remove(f))
      if (maxindex > 0) {

        //val splitFeature = idx(maxindex)

        if (!numIdx.contains(maxindex)) { //连续属性不删，可进一步分裂
          idx.remove(maxindex)
        }
      } else {
        maxindex
      }

      var sets = new HashMap[String, Node]()

      if (idx.size > 0) {
        //对数据进行分裂，递归计算
        var splitData = new HashMap[String, ArrayBuffer[LabeledFeature]]
        data.map(f => {
          var ty = f.features(maxindex)
          //if("".equalsIgnoreCase(ty)){ty = "_NAN_"}
          //f.features.drop(n)
          if (!numIdx.contains(maxindex)) {
            if (!splitData.contains(ty)) { splitData.put(ty, new ArrayBuffer[LabeledFeature]) }
            splitData(ty).+=(f)
          } else {
            ty = if ("?".equalsIgnoreCase(ty)) ty else { if (ty.toDouble > split_point(maxindex)) { ">" + split_point(maxindex) } else { "<=" + split_point(maxindex) } }
            if (!splitData.contains(ty)) { splitData.put(ty, new ArrayBuffer[LabeledFeature]) }
            splitData(ty).+=(f)
          }
        })
        //val miss = splitData.remove("_NAN_")
        val miss = splitData.remove("?")
        splitData = splitData.filter(p => { p._2.size >= 2 }) //每个叶子节点要有两个实例

        if (splitData.size > 1) { //没必要在进行一次递归了

          var f_cnt = new HashMap[String, Int]
          splitData.map(f => {
            val ty = f._1
            val va = f._2

            va.map(l => {
              f_cnt(ty) = f_cnt.getOrElse(ty, 0) + va.size
            })
          })

          var max = -1.0
          var label = ""
          val tt = f_cnt.values.sum
          var f_per = new HashMap[String, Int]
          f_cnt.map(f => {
            if (f._2 > max) { max = f._2; label = f._1 }
            f_per(f._1) = Math.round((f._2 * 1.0f / tt) * miss.size)
          })
          f_cnt.map(f => {
            //if (f._2 > max) { max = f._2; label = f._1 }
            f_per(f._1) = Math.round((f._2 * 1.0f / tt) * miss.size)
          })

          var jj = 0
          splitData.map(f => {
            miss match {
              case None => sets(f._1) = buildDT(f._2, idx, numIdx)
              case Some(t) => sets(f._1) = {
                //              t.map(lb => {
                //                labels(lb.label) = labels(lb.label) + 1
                //              })
                val till = f_per(f._1)

                val slice = t.slice(jj, till)
                jj += till
                val subdata = f._2.++=(slice)
                if (subdata.size > 2) { //避免过度拟合
                  buildDT(subdata, idx, numIdx)
                } else {
                  subtreeNode(subdata)
                }
              } //缺失值按分类数据比例复制到子节点
            }
          })

          //    .map(t=>if(t._2) {splitData.remove(t._1)})
          max = -1.0
          label = ""
          labels.map(f => if (f._2 > max) { max = f._2; label = f._1 })
          val hit = labels(label)
          val mis = labels.values.sum - hit
          //                val hit = sets.values.map(f => f.hit).sum
          //                val mis = sets.values.map(f => f.mis).sum
          //        val hit = label_cnt(label)
          //        val mis = label_cnt.values.sum - hit
          val z = 0.69 //置信度25%=>0.69
          //子树修剪
          //子节点误差率总和
          val esub = sets.values.map(node => {
            val N = node.hit + node.mis
            val f = node.mis * 1.0 / N

            N * 1.0 / (hit + mis) * ((f + z * z / (2 * N) + z * math.sqrt(f / N - f * f / N + z * z / (4 * N * N))) / (1 + z * z / N))
          }).sum
          val N = hit + mis
          val f = mis * 1.0 / N

          //父节点误差率
          val e = ((f + z * z / (2 * N) + z * math.sqrt(f / N - f * f / N + z * z / (4 * N * N))) / (1 + z * z / N))
          if (e < esub) { //符合修剪条件

            Node(-1, -1.0, null, label, hit, mis)
          } else {
            if (numIdx.contains(maxindex)) {
              Node(maxindex, split_point(maxindex), sets, "", hit, mis)
            } else {
              Node(maxindex, -1.0, sets, "", hit, mis)
            }
          }
        } else {
          subtreeNode(data)
        }
      } else { //没有属性分解了
        var max = -1.0
        var index = ""
        labels.map(f => if (f._2 > max) { max = f._2; index = f._1 })
        val hit = labels(index)
        val mis = labels.values.sum - hit
        Node(-1, -1.0, null, index, hit, mis)
      }
    }
  }

  def subtreeNode(subData: ArrayBuffer[LabeledFeature]): Node = {
    var labels = new HashMap[String, Int]()
    subData.map(f => {
      if (!labels.contains(f.label)) labels.+=((f.label, 1)) else labels(f.label) = labels(f.label) + 1
    })
    var max = -1.0
    var index = ""
    labels.map(f => if (f._2 > max) { max = f._2; index = f._1 })
    val hit = labels(index)
    val mis = labels.values.sum - hit
    Node(-1, -1.0, null, index, hit, mis)
  }
  def discrete(data: ArrayBuffer[LabeledFeature],
    //entroy:Double,
    index: Int,
    sorted: TreeSet[Double]): (Double, Double, Double, Double) = {
    //var rs = sorted.
    var arr = sorted.toArray
    println("------------------------------------")
    println("i=" + index + ",sets:" + arr.mkString(","))
    var rs = arr(0)
    var re = arr(0)
    var cu = -1.0
    var back: (Double, Double, Double) = (0.0, 0.0, 0.0)
    //var lastLabel =
    for (i <- 1 until arr.length - 1) {
      //val a1 = arr(i)
      //val a2 = arr(i+1)
      //rs = (a1+a2)/2
      rs = arr(i)
      val cnt = new HashMap[String, HashMap[String, Int]]
      (1 to 2).map(f => cnt("" + f) = new HashMap[String, Int])
      //val entr = new HashMap[String,Double]
      //cnt("?") = new HashMap[String, Int]
      val labels = new HashMap[String, Int]
      var miss_cnt = 0
      data.map(lf => {
        val fff = lf.features(index)
        val label = lf.label
        if (!"?".equalsIgnoreCase(fff)) {
          val v = fff.toDouble

          val n1 = labels.getOrElse(label, 0) + 1
          labels.put(label, n1)
          if (v > rs) {
            val n = cnt("2").getOrElse(label, 0) + 1
            cnt("2").put(label, n)

          } else {
            val n = cnt("1").getOrElse(label, 0) + 1
            cnt("1").put(label, n)

          }
        } else {
          //          val n = cnt("?").getOrElse(label, 0) + 1
          //          cnt("?").put(label, n)
          miss_cnt += 1
        }
      })
      //if(cnt("?").size<1){cnt.remove("?")}
      //val all = cnt.values.map(f => f.values.sum).sum
      //cnt.remove("?")
      val left = cnt.values.map(f => f.values.sum).sum
      val entr = cnt.map(f => {
        val base = f._2.values.sum
        (f._1,
          f._2.map(lb => {
            val p = lb._2 * 1.0 / base
            -1.0 * p * log2(p)
          }).sum)
      })
      val entroy = labels.map(f => {
        val p = 1.0 * f._2 / left
        -1.0 * p * log2(p)
      }).sum
      val info = cnt.map(f => (f._2.values.sum * 1.0 / left) * entr(f._1)).sum
      val gain = entroy - info
      val split_gain = cnt.map(f => { -1 * (f._2.values.sum * 1.0 / left) * log2(f._2.values.sum * 1.0 / left) }).sum
      val dis = (left * 1.0 / (left + miss_cnt))
      val r = gain
      println("info:" + info + ",gain=" + gain + ",split=" + split_gain + ",ratio=" + gain / split_gain + ",re=" + rs)
      if (r >= cu) { cu = r; back = (info, split_gain, dis); re = rs }
    }
    (re, back._1, back._2, back._3)
  }
  def log2(a: Double): Double = {
    math.log(a) / math.log(2)
  }
  def printTree(node: Node, lev: Int) {
    println("->" * lev + "i=" + node.i + ",con=" + node.split + ",hit=" + node.hit + ",mis=" + node.mis + ",class=" + node.label)
    // println("-"*lev + node.toString)
    if (node.sets != null) node.sets.map(n => printTree(n._2, lev + 1))
  }
  def main(args: Array[String]): Unit = {
    var numIdx = new HashSet[Int]
    //        numIdx.+=(0)
    numIdx.+=(1)
    numIdx.+=(2)
    //        numIdx.+=(3)
    //        numIdx.+=(5)
    //        numIdx.+=(7)
    //        numIdx.+=(8)
    //        numIdx.+=(10)
    val insts = new Instances(numIdx)
    insts.read("E:/books/spark/ml/decisionTree/weather.csv")

    val t = classifier(insts)

    printTree(t, 0)
  }

}