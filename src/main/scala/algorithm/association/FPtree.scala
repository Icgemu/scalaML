package algorithm.association

import scala.collection.mutable.HashMap
import scala.io.Source
import java.io.File
import scala.collection.mutable.HashSet
import scala.collection.mutable.ArrayBuffer

object FPtree {

  def find_frequent(file: String, confidence: Double, support: Int) {
    var buff = Source.fromFile(new File(file))
    var itemsets = new HashMap[String, Int]
    val lines = buff.getLines

    for (line <- lines) {
      val arr = line.split(",")
      arr.tail.map(f => {
        val itemset = f
        itemsets(itemset) = itemsets.getOrElse(itemset, 0) + 1
      })
    }
    buff.close
    var header = itemsets.filter(p => { p._2 >= support }).toArray
    header = header.sortBy(f => { f._2 })

    //项头表
    val table = header.map(f => {
      (f._1, new Header(f._1, null, f._2))
    })
    //事务项通过这个map排序
    val hMap = table.toMap

    //root
    val root = new Tree(null, 0, new HashMap[String, Tree], null, null)

    val tbuff = Source.fromFile(new File(file))
    val ls = tbuff.getLines
    for (line <- ls) {
      val arr = line.split(",")
      val t = arr.tail.map(f => {
        val itemset = hMap(f).count
        (f, itemset)
      })
      //按照项头表顺序对改事务项排序
      val s = t.sortBy(f => {
        f._2
      }).reverse.map(f => f._1)
      insert(root, s, hMap, 1)
    }
    tbuff.close
    //插入数据完成
    //fp-growth开始

    val Arr = HashMap[HashSet[String], Int]()
    FpGrowth(root, ArrayBuffer[String](), Arr, hMap)

    Arr.map(f => {
      println(f._1.mkString("&&") + "=>" + f._2)
    })
    //输出关联规则及置信度
    val fre = Arr
    fre.map(t => {
      //f.map(t => {
      if (t._1.size > 1) {
        val sets = Apriori.subsets(t._1)
        for (i <- 0 until sets.size) {
          val it1 = sets(i)

          for (j <- 0 until sets.size) {
            val it2 = sets(j)
            val inter = it1.intersect(it2)
            if (inter.size == 0) {
              val a = Arr(t._1)
              val b = Arr(it2)
              val conf = (a * 1.0 / b)
              if (conf > confidence) {
                println(it2.mkString("&&") + "=>" + it1.mkString("&&") +
                  " ,confidence=" + a + "/" + b + "=" + conf.toString())
              }
            }
          }
        }
      }
    })
    //
  }
  def FpGrowth(tree: Tree,
    base: ArrayBuffer[String],
    hMap: HashMap[HashSet[String], Int], table: Map[String, Header]) {

    //
    if (isSinglePath(tree)) {
      var support = Int.MaxValue
      var node = tree.nodes.head
      val set = ArrayBuffer[String]()
      while (node != null) {
        set += (node._1)
        if (node._2.count < support) {
          support = node._2.count
        }
        if (node._2.nodes.size > 0) node = node._2.nodes.head else {
          node = null
        }
      }
      val subsets = Apriori.subsets(HashSet(set.toArray: _*))

      for (s <- subsets.toIterator) {
        val ss = s.union(HashSet(base.toArray: _*))
        hMap(ss) = support
      }

    } else {
      //按项头集的support 由小到大，构造模式条件基，递归调用FpGrowth
      var n = table.map(f => {
        (f._1, f._2.count)
      }).toArray
      n = n.sortBy(f => f._2)
      val nodes = n.map(f => { f._1 })

      for (a <- nodes.toIterator) {
        val ai = a
        val bs = base.map(f => f)
        bs += (ai)
        val ss = HashSet(bs: _*)

        hMap(ss) = hMap.getOrElse(ss, 0) + table(ai).count

        val tr = modelBase(tree, ai, table)
        tr match {
          case None =>
          case Some(t) => {
            if (t._1.nodes.size > 0) FpGrowth(t._1, bs, hMap, t._2)
          }
        }
      }
    }
  }
  //b的条件模式基
  def modelBase(tree: Tree,
    b: String,
    table: Map[String, Header]): Option[(Tree, Map[String, Header])] = {
    val Arr = ArrayBuffer[(ArrayBuffer[String], Int)]()
    val t = tree.nodes.map(f => f)
    var header = table(b).header
    //找到b的所有节点，查找它们到root节点的路径
    while (header != null) {
      val su = ArrayBuffer[String]()
      var p = header.parent
      val c = header.count
      while (p.name != null) {
        su += (p.name)
        p = p.parent
      }
      if (su.size > 0) {
        Arr += ((su, c))
      }
      header = header.next
    }
    //如果路径不为空，把路径看做数据项，数据项的support为b对应节点的support,重新构造新树
    if (Arr.size > 0) {

      val root = new Tree(null, 0, new HashMap[String, Tree], null, null)

      var itemsets = new HashMap[String, Int]
      for (line <- Arr.toIterator) {
        val arr = line._1
        arr.map(f => {
          val itemset = f
          itemsets(itemset) = itemsets.getOrElse(itemset, 0) + line._2
        })
      }
      //var header = itemsets.filter(p => { p._2 >= 2 }).toArray
      var header = itemsets.toArray.sortBy(f => { f._2 })

      //项头表
      val table = header.map(f => {
        (f._1, new Header(f._1, null, f._2))
      })
      //事务项通过这个map排序
      val hMap = table.toMap

      for (line <- Arr.toIterator) {
        val arr = line._1
        val t = arr.map(f => {
          val itemset = hMap(f).count
          (f, itemset)
        })
        //按照项头表顺序对改事务项排序
        val s = t.sortBy(f => {
          f._2
        }).reverse.map(f => f._1)
        insert(root, s.toArray, hMap, line._2)
      }

      //删掉置信度低的
      prune(root)

      if (root.nodes.size > 0) {
        Option((root, hMap))
      } else {
        None
      }
    } else {
      None
    }
  }

  //裁剪support<2的节点
  def prune(tree: Tree) {
    val s = tree.nodes.keys
    s.map(f => {
      if (tree.nodes(f).count < 2) {
        tree.nodes.remove(f)
      } else {
        prune(tree.nodes(f))
      }
    })
  }

  //是否是单路径，即各节点最多只有一个子节点
  def isSinglePath(tree: Tree): Boolean = {

    var nodes = tree.nodes
    var is = true
    while (is && nodes.size > 0) {
      if (nodes.size > 1) {
        is = false
      } else {
        val t = nodes.toArray
        val t2 = t(0)._1
        nodes = nodes(t2).nodes
      }
    }
    is
  }

  //插入一个support为CNt的项
  def insert(root: Tree, items: Array[String], hMap: Map[String, Header], cnt: Int) {
    if (items != null && items.size > 0) {
      if (root.nodes.contains(items(0))) {
        root.nodes(items(0)).count += cnt
      } else {
        val n = new Tree(items(0), cnt, new HashMap[String, Tree], null, root)
        root.nodes(items(0)) = n
        val h = hMap(items(0)).header
        if (h == null) {
          hMap(items(0)).header = n
        } else {
          var h = hMap(items(0)).header
          while (h.next != null) {
            h = h.next
          }
          h.next = n
        }
      }

      insert(root.nodes(items(0)), items.tail, hMap, cnt)
    }
  }

  def main(args: Array[String]): Unit = {
    find_frequent("E://books//spark//ml//Apriori//data.csv", 0.0, 1)
  }

}