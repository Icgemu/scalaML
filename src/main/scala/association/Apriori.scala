package association

import scala.io.Source
import java.io.File
import scala.collection.mutable.HashMap
import scala.collection.mutable.HashSet
import scala.collection.mutable.ArrayBuffer

//http://rakesh.agrawal-family.com/papers/vldb94apriori.pdf
//Fast Algorithms for Mining Association Rules
object Apriori {

  def find_frequent(file: String, confidence: Double, support: Int) {
    var buff = Source.fromFile(new File(file))
    var itemsets = new HashMap[HashSet[String], Int]
    val D = new HashMap[HashSet[String], Int]
    buff.getLines.map(line => {
      val arr = line.split(",")
      arr.tail.map(f => {
        val itemset = HashSet[String](f)
        itemsets(itemset) = itemsets.getOrElse(itemset, 0) + 1

      })
      val itemset = HashSet[String](arr.tail: _*)
      D(itemset) = D.getOrElse(itemset, 0) + 1
    })
    buff.close
    itemsets = itemsets.filter(p => { p._2 >= support })
    val Arr = ArrayBuffer[HashMap[HashSet[String], Int]]()
    Arr += itemsets
    //递归查找频繁集
    frequent(itemsets, D, support, Arr)

    //输出关联规则及置信度
    val fre = Arr.toArray
    fre.tail.map(f => {

      f.map(t => {
        val sets = subsets(t._1)

        for (i <- 0 until sets.size) {
          val it1 = sets(i)

          for (j <- 0 until sets.size) {
            val it2 = sets(j)
            val inter = it1.intersect(it2)
            if (inter.size == 0) {
              val a = Arr(t._1.size - 1)(t._1)
              val b = Arr(it2.size - 1)(it2)
              val conf = (a * 1.0 / b)
              if (conf > confidence) {

                println(it2.mkString("&&") + "=>" + it1.mkString("&&") +
                  " ,confidence=" + a + "/" + b + "=" + conf.toString())
              }
            }
          }
        }
      })
    })
    //
  }
  def subsets(set: HashSet[String]): Array[HashSet[String]] = {
    val arr = HashSet[HashSet[String]]()
    set.map(f => { arr.+=(HashSet(f)) })
    sub(arr)
    arr.toArray
  }
  def sub(set: HashSet[HashSet[String]]) {
    val s = set.map(f => f)
    val m = HashSet[HashSet[String]]()
    s.map(f => {
      s.map(t => {
        val inter = f.intersect(t)
        if (inter.size == 0) {
          val u = t.union(f)
          if (!set.contains(u)) {
            m.+=(u)
          }
        }
      })
    })
    if (m.size > 0) {
      set.++=(m)
      sub(set)
    }
  }
  def frequent(itemsets: HashMap[HashSet[String], Int],
    D: HashMap[HashSet[String], Int],
    support: Int,
    Arr: ArrayBuffer[HashMap[HashSet[String], Int]]) {

    //连接步
    val keys = itemsets.keys
    val Ck = new HashMap[HashSet[String], Int]
    for (key <- keys) {
      val oldcnt = itemsets(key)
      itemsets.remove(key)
      val ks = itemsets.keys
      for (k <- ks) {
        val bool = link(key, k)
        if (bool) {
          val set = key.union(k)
          if (!Ck.contains(set)) {
            Ck(set) = 0
          }
        }
      }
      itemsets(key) = oldcnt
    }

    //剪枝步
    var ks = Ck.keys

    for (k <- ks) {
      val arr = k.toArray
      arr.map(item => {
        k.remove(item)
        if (!itemsets.contains(k)) {
          Ck.remove(k)
        }
        k.+=(item)
      })
    }

    //搜索Ck个元素的support
    ks = Ck.keys
    for (k <- ks) {

      D.map(item => {
        val set = item._1.map(f => f)
        if (set.size >= k.size) {
          val s = set.size
          val u = set.union(k)
          if (u.size == s) {
            Ck(k) = Ck(k) + D(item._1)
          }
        }
      })
    }
    val Ik = Ck.filter(p => { p._2 >= support })
    if (Ik.size > 0) Arr += Ik
    //尾递归
    if (Ik.size > 0) frequent(Ik, D, support, Arr)
  }

  def link(a: HashSet[String], b: HashSet[String]): Boolean = {
    val a_ = a.toArray.slice(0, a.size - 1)
    val b_ = b.toArray.slice(0, b.size - 1)
    var bool = true
    a_.zip(b_).map(t => { if (!t._1.equalsIgnoreCase(t._2)) { bool = false } })
    bool
  }

  def main(args: Array[String]): Unit = {
    find_frequent("E://books//spark//ml//Apriori//data.csv", 0.1, 1)
  }

}