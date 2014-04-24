package classifier.tree

import classifier.Model
import core.Instances
import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuffer
import core.Feature
import scala.collection.mutable.HashSet
import scala.collection.mutable.TreeSet

class C45(insts: Instances) extends TreeClassifierBase {

  def train(): Model = {
    val data = insts.data
    val idx = insts.idxForNominal
    val numIdx = insts.numIdx

    C45Model(train0(data, idx, numIdx))
  }

  def train0(
    data: ArrayBuffer[Feature],
    idx: HashMap[Int, HashSet[String]],
    numIdx: HashSet[Int]): Node = {

    val split_point = new HashMap[Int, Double]
    val d_ratio = ratio(data)
    val filters = d_ratio.filter(t => t._2 > 0.95).toArray

    if (filters.size > 0) {
      val (hit, mis) = hitAndMiss(data, filters(0)._1)
      Node(-1, -1.0, null, filters(0)._1, hit, mis)
    } else {
      val e_entropy = exp_entropy(data)

      val c_entropy = new HashMap[Int, Double]
      val h_entropy = new HashMap[Int, Double]
      val miss_discount_ratio = new HashMap[Int, Double]

      var remove_1 = new HashSet[Int]

      for (i <- idx.keys) {
        val s = idx(i)
        if (s.size > 0) { //nominal
          val sets = data.groupBy(f => f.features(i))
          val miss = sets.get("?") //miss value
          miss_discount_ratio(i) = miss match {
            case None => 1.0 // no miss value,discount ratio is 1.0
            case Some(t) => t.map(f => f.weight).sum / data.map(f => f.weight).sum
          }
          //val notmissset = sets.filterKeys(p=>p.equalsIgnoreCase("?"))
          val sub = data.filterNot(p => p.features(i).equals("?")).groupBy(f=>f.features(i))
          c_entropy(i) = cond_entropy(sub)
          h_entropy(i) = ha_entropy(sub)
        } else {

          var sorted = new TreeSet[Double]()
          var nosorted = new TreeSet[Double]()
          var lastLabel = data(0).target
          var lastV = -1.0
          val sortmap = data.sortBy(f => { f.features(i) })

          sortmap.map(lf => {
            val v = lf.features(i)
            if (!lf.target.equals(lastLabel) && !"?".equalsIgnoreCase(v)) {
              sorted += v.toDouble
            }
            if (!"?".equalsIgnoreCase(v)) {
              lastLabel = lf.target
              lastV = v.toDouble
              nosorted += lastV
            }
          })
          //寻找信息值最大的分裂点
          if (sorted.size > 1) {
            val p = discrete(data, i, sorted)
            c_entropy(i) = p._2
            h_entropy(i) = p._3
            split_point(i) = p._1
            miss_discount_ratio(i) = p._4
          } else {
            remove_1 += i
          }
        }
      }

      //找出最大信息增益率
      //找出增益均值以上的才进行计算增益率，避免数据偏移比较严重的分裂值很小
      val avg = c_entropy.map(f => { (e_entropy - f._2) * miss_discount_ratio(f._1) }).sum / c_entropy.size
      var gainRatio = c_entropy.filter(p => {
        (e_entropy - p._2) * miss_discount_ratio(p._1) >= avg
      }).map(f => {
        (f._1, (e_entropy - f._2) * miss_discount_ratio(f._1) / h_entropy(f._1))
      })

      var maxGainRatio = (-1.0)
      var maxGainRatioIndex = (-1)
      for (i <- gainRatio.keys) {
        if (gainRatio(i) > maxGainRatio) { maxGainRatio = gainRatio(i); maxGainRatioIndex = i }
      }

      remove_1.map(f => idx.remove(f))

      if (maxGainRatioIndex > -1) {
        if (!numIdx.contains(maxGainRatioIndex)) { //numerical value can split again
          idx.remove(maxGainRatioIndex)
        }
      } else {
        idx.clear
      }

      var sets = new HashMap[String, Node]()

      if (idx.size > 0) {
        //对数据进行分裂，递归计算
        var splitData = new HashMap[String, ArrayBuffer[Feature]]
        data.map(f => {
          var ty = f.features(maxGainRatioIndex)
          if (!numIdx.contains(maxGainRatioIndex)) {
            if (!splitData.contains(ty)) {
              splitData.put(ty, new ArrayBuffer[Feature])
            }
            splitData(ty) += f
          } else {
            ty = if ("?".equalsIgnoreCase(ty)) ty else {
              if (ty.toDouble > split_point(maxGainRatioIndex)) {
                //">" + split_point(maxGainRatioIndex)
                "2"
              } else {
                //"<=" + split_point(maxGainRatioIndex)
                "1"
              }
            }
            if (!splitData.contains(ty)) { splitData.put(ty, new ArrayBuffer[Feature]) }
            splitData(ty) += f
          }
        })

        val miss = splitData.remove("?")

        splitData = splitData.filter(p => { p._2.size >= 2 }) //每个叶子节点要有两个实例

        if (splitData.size > 1) { //没必要在进行一次递归了

          var f_cnt =splitData.map(f=>{(f._1,f._2.size)})

          val tol = f_cnt.values.sum
          var f_per = f_cnt.map(f => {
            (f._1,Math.round((f._2 * 1.0f / tol) * miss.size))
          })

          var jj = 0
          splitData.map(f => {
            val f1 = f._1
            val f2 = f._2
            miss match {
              case None => sets(f1) = train0(f2, idx, numIdx)
              case Some(t) => sets(f1) = {
                val till = f_per(f1)

                val slice = t.slice(jj, jj+till)
                jj += till
                val subdata = f2 ++ slice
                if (subdata.size > 2) { //避免过度拟合
                  train0(subdata, idx, numIdx)
                } else {
                  makeLeafNode(subdata)
                }
              } //缺失值按分类数据比例复制到子节点
            }
          })

          val sorted = d_ratio.toArray.sortBy(f => f._2).reverse
          val (hit, mis) = hitAndMiss(data, sorted(0)._1)

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
            Node(-1, -1.0, null, sorted(0)._1, hit, mis)
          } else {
            if (numIdx.contains(maxGainRatioIndex)) {
              Node(maxGainRatioIndex, split_point(maxGainRatioIndex), sets, sorted(0)._1, hit, mis)
            } else {
              Node(maxGainRatioIndex, -1.0, sets, sorted(0)._1, hit, mis)
            }
          }
        } else {
          makeLeafNode(data)
        }

      } else {
        val sorted = d_ratio.toArray.sortBy(f => f._2).reverse
        val (hit, mis) = hitAndMiss(data, sorted(0)._1)
        Node(-1, -1.0, null, sorted(0)._1, hit, mis)
      }
    }
  }

  def makeLeafNode(subData: ArrayBuffer[Feature]): Node = {
    val r = ratio(subData)
    val sorted = r.toArray.sortBy(f => f._2).reverse
    val (hit, mis) = hitAndMiss(subData, sorted(0)._1)
    Node(-1, -1.0, null, sorted(0)._1, hit, mis)
  }

  def discrete(
    data: ArrayBuffer[Feature],
    index: Int,
    sorted: TreeSet[Double]): (Double, Double, Double, Double) = {
    var arr = sorted.toArray
    //println("------------------------------------")
    //println("i=" + index + ",sets:" + arr.mkString(","))
    var rs = arr(0)
    var re = arr(0)
    var cu = -1.0
    var back: (Double, Double, Double) = (0.0, 0.0, 0.0)

    for (i <- 1 until arr.length - 1) {
      rs = arr(i);
      val miss_cnt = data.filter(p => p.features(index).equals("?")).map(f => f.weight).sum
      val sets = data.filterNot(p => p.features(index).equals("?")).groupBy(f=>if(f.features(index).toDouble>rs)"1"else"0")

      val info = cond_entropy(sets)
      val gain = gain_entropy(sets)
      val h_gain = ha_entropy(sets)

      val tol = sets.values.flatMap(f => f.map(lf=>lf.weight)).sum
      val dis = (tol / (tol + miss_cnt))

      //println("info:" + info + ",gain=" + gain + ",split=" + h_gain + ",ratio=" + gain / h_gain + ",re=" + rs)
      if (gain >= cu) { cu = gain; back = (info, h_gain, dis); re = rs }

    }
    (re, back._1, back._2, back._3)
  }

  case class C45Model(nodes: Node) extends Model {
    def predict(test: Instances): Double = {
      var r = 0.0
      test.data.map(lf => {
        var node = nodes
        var label = ""
        var isbreak = false
        while (node.i > 0 && !isbreak) {
          label = node.label
          val f = lf.features(node.i)
          if (f.equals("?")) {
            isbreak = true
          } else {
            val n = if (node.split == -1) {
              node.sets.get(f)
            } else {
              node.sets.get(if (f.toDouble > node.split) "2" else "1")
            }
            n match {
              case None => isbreak = true
              case Some(t)=>node = t
            }
          }
        }
        label = node.label
        if (label.equalsIgnoreCase(lf.target)) r += 1.0

      })
      r / test.data.size
    }
  }
  case class Node(i: Int, split: Double, sets: HashMap[String, Node], label: String, hit: Double, mis: Double)
}

object C45 {

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

    val t = new C45(trainset)
    val model = t.train()

    val accu = model.predict(testset)
    println(accu);
  }
}