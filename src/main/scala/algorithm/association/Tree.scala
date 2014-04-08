package algorithm.association

import scala.collection.mutable.HashMap

class Tree(n: String, c: Int, ns: HashMap[String, Tree], ne: Tree,p:Tree) {
    val name = n
    var count = c
    val nodes = ns
    var next = ne
    var parent = p
  }