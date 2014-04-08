package algorithm.tree

import algorithm.RegInstances
import scala.collection.mutable.HashMap
import scala.collection.mutable.HashSet
import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import java.io.File
import algorithm.RegFeature

object DecisionStump {

  class Node(var index: Int, var split: String, var i: Int,
    var left: Int, var right: Int,
    var parent: Int, var err: Double,
    var c: Double, var miss: Double, var size: Int, var leaf: Int) {

  }

  var ite = 0
  val N = 1
  def classifier(insts: RegInstances, J: Int): HashMap[Int, Node] = {
    var nodes = HashMap[Int, Node]()
    val data = insts.data
    val idx = insts.idx
    val numIdx = insts.numIdx
    buildDT(nodes, data, idx, numIdx, 1)

    get(nodes, J)
    //prune()
    //printTree(nodes(1), 0)
    //println("--------------")
    nodes
  }
  def get(nodes: HashMap[Int, Node], J: Int) {
    val set = HashSet(2, 3)
    val j = if (J % 2 == 0) J else J + 1
    val okset = HashSet[Int]()
    while ((set.size + okset.size) < j && set.size > 0) {
      val sl = set.toArray.sortBy(f => f).first
      val l = sl * 2
      val r = l + 1
      set.remove(sl)
      if (nodes.contains(l)) {
        set.+=(l)
        set.+=(r)
      } else {
        okset += sl
      }
    }
    okset ++= set
    //printTree(nodes,nodes(1), 0)
    //println(okset.mkString(","))
    val l = nodes(okset.toArray.sortBy(f => f).last).index
    truncateNode(nodes, nodes(1), l)
    pruneNode(nodes, nodes(1))
    val remove = nodes.filter(p => {
      p._1 > l
    })
    remove.map(f => {
      nodes.remove(f._1)
    })
    //nodes = nodes.map(f=>f)
  }

  def prune(nodes: HashMap[Int, Node]) {
    val notLeaf = nodes.filter(p => p._2.leaf > 1)
    var minerr = Double.MaxValue
    var minIndex = -1
    var err = notLeaf.map(node => {

      val n = node._2
      val NTt = n.leaf
      val RTt = n.err
      val Rt = n.c
      val alpha = (Rt - RTt) * 1.0 / (math.abs(NTt) - 1)
      if (alpha < minerr) { minerr = alpha; minIndex = node._1 }
    })

    if (minIndex % 2 == 0) { minIndex += 1 }
    truncateNode(nodes, nodes(1), minIndex)
    pruneNode(nodes, nodes(1))
    val remove = nodes.filter(p => {
      p._1 > minIndex
    })
    remove.map(f => {
      nodes.remove(f._1)
    })
  }

  def truncateNode(nodes: HashMap[Int, Node], node: Node, maxIndex: Int) {
    if (node.index > maxIndex) { nodes.remove(node.index) }
    else {
      if (node.left > 0) truncateNode(nodes, nodes(node.left), maxIndex)
      if (node.right > 0) truncateNode(nodes, nodes(node.right), maxIndex)
    }
  }
  def pruneNode(nodes: HashMap[Int, Node], node: Node) {
    if (nodes.contains(node.left)) { pruneNode(nodes, nodes(node.left)); pruneNode(nodes, nodes(node.right)) }
    else {
      nodes(node.index) = new Node(node.index, node.split, node.i, -1, -1, node.parent,
        node.err, node.c, node.miss, node.size, node.leaf)
    }
  }

  def buildDT(nodes: HashMap[Int, Node], data: ArrayBuffer[RegFeature],
    idx1: HashMap[Int, HashSet[String]],
    numIdx: HashSet[Int], index: Int) {

    //var iClass = data.groupBy(f => f.label).map(f => (f._1, f._2.size))
    //val classSum = iClass.values.sum
    //val classFraction = iClass.map(f => (f._1, f._2 * 1.0 / classSum))
    //val isLeaf = classFraction.filter(p => p._2 > 0.95).size > 0
    //val isLeaf = data.size == 1
    //println(data.size)
    //if (!isLeaf) {

    val keyMaxSize = idx1.keys.max + 1
    var gini = Array.fill(keyMaxSize)(0.0)
    var split = Array.fill(keyMaxSize)("")
    var dist = Array.fill(keyMaxSize)(Array.fill(2)(0.0))
    var fration = Array.fill(keyMaxSize)(Array.fill(2)(0.0))
    var dataLeft = Array.fill(keyMaxSize)(ArrayBuffer[RegFeature]())
    var dataRight = Array.fill(keyMaxSize)(ArrayBuffer[RegFeature]())
    var dataMiss = Array.fill(keyMaxSize)(ArrayBuffer[RegFeature]())
    for (iAttr <- idx1.keys) {
      val isNumeric = numIdx.contains(iAttr)
      if (isNumeric) {

        var bestSplit = splitNumeric(index, data, iAttr)
        split(iAttr) = bestSplit._1 + ""
        gini(iAttr) = bestSplit._2
        dist(iAttr) = bestSplit._3
        fration(iAttr) = bestSplit._4
        dataLeft(iAttr) = bestSplit._5
        dataRight(iAttr) = bestSplit._6
        dataMiss(iAttr) = bestSplit._7
      } else {
        var bestSplit = splitNonimal(data, iAttr)
        split(iAttr) = bestSplit._1
        gini(iAttr) = bestSplit._2
        dist(iAttr) = bestSplit._3
        fration(iAttr) = bestSplit._4
        dataLeft(iAttr) = bestSplit._5
        dataRight(iAttr) = bestSplit._6
        dataMiss(iAttr) = bestSplit._7
      }
    }

    //println(index+"=>"+gini.mkString(","))
    //println("------------")
    val bestAttr = minIndex(gini)
    val isNumeric = numIdx.contains(bestAttr)
    val bestDist = dist(bestAttr)
    val bestSplit = split(bestAttr)
    var left = dataLeft(bestAttr)
    var right = dataRight(bestAttr)
    var miss = dataMiss(bestAttr)

    val lc = valuefor(left)
    val mc = valuefor(miss)
    val rc = valuefor(right)
    val ac = valuefor(data)
    println("miss=>"+miss.size)
    nodes(index * 2) = new Node(index * 2, "", -1, -1, -1, index, lc, lc, 0.0, left.size, 1)
    nodes(index * 2 + 1) = new Node(index * 2 + 1, "", -1, -1, -1, index, rc, rc, 0.0, right.size, 1)
    nodes(index) = new Node(index, bestSplit, bestAttr, 2 * index, 2 * index + 1, -1, lc + rc, ac, mc, data.size, 2)

    //println(left.size +"=>" + right.size)
    //      if (left.size > N && right.size > N) {
    //        val leftNode = {
    //          var d = idx1.map(f => f)
    //          if (!isNumeric) d(bestAttr).remove(bestSplit)
    //          val node = buildDT(nodes,left, d, numIdx, 2 * index)
    //          nodes(2 * index) = node
    //          node
    //        }
    //        val rightNode = {
    //          var d = idx1.map(f => f)
    //          if (!isNumeric) { d(bestAttr).remove(bestSplit) }
    //          val node = buildDT(nodes,right, d, numIdx, 2 * index + 1)
    //          nodes(2 * index + 1) = node
    //          node
    //        }
    //        val parent = if (index % 2 == 0) { index / 2 } else { (index - 1) / 2 }
    //        val NTt = leftNode.leaf + rightNode.leaf
    //
    //        val l = valuefor(data)
    //        val miss = valuefor(data)
    //        val node = new Node(index,
    //          split(bestAttr),
    //          bestAttr,
    //          2 * index, 2 * index + 1, parent,
    //          leftNode.err + rightNode.err,
    //          l, data.size,
    //          NTt)
    //        nodes(index) = node
    //        node
    //      } else {
    //        val parent = if (index % 2 == 0) { index / 2 } else { (index - 1) / 2 }
    //        val l = valuefor(data)
    //        val node = new Node(index, "", -1, -1, -1, parent, l, l, data.size, 1)
    //        nodes(index) = node
    //       // println(node.c)
    //        node
    //      }

    //    } else {
    //      val parent = if (index % 2 == 0) { index / 2 } else { (index - 1) / 2 }
    //      val l = valuefor(data)
    //      val node = new Node(index, "", -1, -1, -1, parent, l, l, data.size, 1)
    //      nodes(index) = node
    //      //println(node.c)
    //      node
    //    }
  }
  //  def labelfor(data: ArrayBuffer[LabeledFeature]): (String, Int, Int) = {
  //
  //    var iClass = data.groupBy(f => f.label).map(f => (f._1, f._2.size))
  //    val classSum = iClass.values.sum
  //    val classFraction = iClass.map(f => (f._1, f._2 * 1.0 / classSum))
  //    var max = -1.0
  //    var label = ""
  //    classFraction.map(c => { if (c._2 > max) { max = c._2; label = c._1 } })
  //    val hit = data.filter(d => d.label.equalsIgnoreCase(label)).size
  //    val mis = data.size - hit
  //    (label, hit, mis)
  //  }

  def valuefor(data: ArrayBuffer[RegFeature]): Double = {

    val sum1 = data.map(f => f.value * f.weight).sum
    val w = data.map(f => f.weight).sum
    //val sum3 = data.map(f =>  (1 - math.abs(f.value))).sum
    //val sum2 = data.map(f => math.abs(f.value) * (1 - math.abs(f.value))).sum
    // val avgright = right.map(f=>f.value).sum/right.size
    //val loss = data.map(f => (f.value - avg) * (f.value - avg)).sum
    //val lossright = left.map(f=>(f.value-avgright)*(f.value-avgright)).sum
    //loss
    //    println(sum1+"=>"+sum2+"="+(sum1 / sum2))
    //val t = if(sum1<0) -1.0 else 1.0
    //    if(math.abs(sum3)<1e-10){
    //      100
    //    }else{
    //      sum1 / sum2
    //    }
    sum1 / w

  }
  def minIndex(Arr: Array[Double]): Int = {
    var i = 0
    var min = Arr(0)
    var j = 0
    Arr.map(f => { if (f < min) { min = f; i = j }; j += 1 })
    i
  }
  def splitNonimal(instances: ArrayBuffer[RegFeature],
    iAttr: Int): (String, Double, Array[Double], Array[Double], ArrayBuffer[RegFeature], ArrayBuffer[RegFeature], ArrayBuffer[RegFeature]) = {
    var bestCutPoint = ""
    var bestGini = Double.MaxValue
    var bestDist = Array.fill(2)(0.0)
    var bestFaction = Array.fill(2)(0.0)
    val missInsts = instances.filter(p => p.features(iAttr).equalsIgnoreCase("?"))
    val hasValueInsts = instances.filter(p => { !p.features(iAttr).equalsIgnoreCase("?") })
    val allValue = hasValueInsts.map(f => f.features(iAttr)).toSet.toArray

    var bestLeft = ArrayBuffer[RegFeature]()
    var bestRight = ArrayBuffer[RegFeature]()
    var bestMiss = missInsts
    //val parentDist = instances.groupBy(f => f.label).map(f => (f._1, f._2.size * 1.0))

    for (currCutPoint <- allValue.toIterator) {

      //val currCutPoint = (sortedValue(i) + sortedValue(i+1))/2.0
      var dist = Array.fill(2)(0.0)

      val left = ArrayBuffer[RegFeature]()
      val right = ArrayBuffer[RegFeature]()
      val miss = missInsts.map(f => f)
      hasValueInsts.map(inst => {
        val f = inst.features(iAttr)
        if (!currCutPoint.equalsIgnoreCase(f)) {
          val t = dist(1)
          dist(1) = t + inst.weight
          right += inst
        } else {
          dist(0) = dist(0) + inst.weight
          left += inst
        }
      })

      //val tempDist = Array.fill(2)(0.0)
      //tempDist(0) = dist(0).values.sum * 1.0
      //tempDist(1) = dist(1).values.sum * 1.0

      val tempSum = dist.sum
      val tempPro = dist.map(t => t * 1.0 / tempSum)

      //      var w = miss.map(f=>f.weight).sum
      //      val leftsize = (tempPro(0) * miss.size).toInt
      //      for (i <- 0 until leftsize) {
      //        val t = (math.random * miss.size).toInt
      //        val tt = miss.remove(t)
      //        left += tt
      //        dist(0) += tt.weight
      //        w -= tt.weight
      //      }
      //      right ++= miss
      //      dist(1) += w

      val avgleft = left.map(f => f.value * f.weight).sum
      val avgright = right.map(f => f.value * f.weight).sum
      val avgmiss = miss.map(f => f.value * f.weight).sum
      
      val sqrleft = left.map(f => (f.value) * (f.value) * f.weight).sum
      val sqrright = right.map(f => (f.value) * (f.value) * f.weight).sum
      val sqrmiss = miss.map(f => (f.value) * (f.value) * f.weight).sum
      
      val swleft = left.map(f => f.weight).sum
      val swright = right.map(f => f.weight).sum
      val swmiss = miss.map(f => f.weight).sum
      
      val lossmiss = sqrmiss - avgmiss * avgmiss / swmiss
      val lossleft = sqrleft - avgleft * avgleft / swleft
      val lossright = sqrright - avgright * avgright / swright
      
      val currGini = lossleft + lossright + lossmiss //computeGiniGain(dist, parentDist)
      if (currGini < bestGini && left.size > N && right.size > N) {
        bestGini = currGini
        bestCutPoint = currCutPoint
        bestDist = dist
        bestFaction = tempPro
        bestLeft = left
        bestRight = right
      }

    }
    (bestCutPoint, bestGini, bestDist, bestFaction, bestLeft, bestRight, bestMiss)
  }
  def splitNumeric(index: Int, instances: ArrayBuffer[RegFeature],
    iAttr: Int): (Double, Double, Array[Double], Array[Double], ArrayBuffer[RegFeature], ArrayBuffer[RegFeature], ArrayBuffer[RegFeature]) = {
    var bestCutPoint = -1.0
    var bestGini = Double.MaxValue
    var bestDist = Array.fill(2)(0.0)
    var bestFaction = Array.fill(2)(0.0)
    val missInsts = instances.filter(p => p.features(iAttr).equalsIgnoreCase("?"))
    val hasValueInsts = instances.filter(p => { !p.features(iAttr).equalsIgnoreCase("?") })
    val allValue = hasValueInsts.map(f => f.features(iAttr).toDouble).toSet.toArray
    val sortedValue = allValue.sortBy(f => f)

    var bestLeft = ArrayBuffer[RegFeature]()
    var bestRight = ArrayBuffer[RegFeature]()
    var bestMiss = missInsts
    //val parentDist = instances.groupBy(f => f.label).map(f => (f._1, f._2.size * 1.0))

    val parentReg = instances.map(f => f.value).sum / instances.size

    if (sortedValue.size < 2) {

    } else {
      for (i <- 0 until sortedValue.length - 1) {
        if (index == 8) {
          print("")
        }
        val currCutPoint = (sortedValue(i) + sortedValue(i + 1)) / 2.0
        var dist = Array.fill(2)(0.0)

        val left = ArrayBuffer[RegFeature]()
        val right = ArrayBuffer[RegFeature]()
        val miss = missInsts.map(f => f)
        hasValueInsts.map(inst => {
          val f = inst.features(iAttr).toDouble
          if (f >=currCutPoint) {
            val t = dist(1)
            dist(1) = t + inst.weight
            right += inst
          } else {
            dist(0) = dist(0) + inst.weight
            left += inst
          }
        })

        //tempDist(0) = dist(0).values.sum * 1.0
        //tempDist(1) = dist(1).values.sum * 1.0

        val tempSum = dist.sum
        val tempPro = dist.map(t => t * 1.0 / tempSum)

        //       var w = miss.map(f=>f.weight).sum
        //      val leftsize = (tempPro(0) * miss.size).toInt
        //      for (i <- 0 until leftsize) {
        //        val t = (math.random * miss.size).toInt
        //        val tt = miss.remove(t)
        //        left += tt
        //        dist(0) += tt.weight
        //        w -= tt.weight
        //      }
        //      right ++= miss
        //      dist(1) += w

        val avgleft = left.map(f => f.value * f.weight).sum
        val avgright = right.map(f => f.value * f.weight).sum
        val avgmiss = miss.map(f => f.value * f.weight).sum
        
        val sqrleft = left.map(f => (f.value) * (f.value) * f.weight).sum
        val sqrright = right.map(f => (f.value) * (f.value) * f.weight).sum
        val sqrmiss = miss.map(f => (f.value) * (f.value) * f.weight).sum
        
        val swleft = left.map(f => f.weight).sum
        val swright = right.map(f => f.weight).sum
        val swmiss = miss.map(f => f.weight).sum
        
        val lossmiss = sqrmiss - avgmiss * avgmiss / swmiss
        val lossleft = sqrleft - avgleft * avgleft / swleft
        val lossright = sqrright - avgright * avgright / swright
        
        val currGini = lossleft + lossright + lossmiss //computeGiniGain(dist, parentDist)

        if (currGini < bestGini && left.size > N && right.size > N) {
          bestGini = currGini
          bestCutPoint = currCutPoint
          bestDist = dist
          bestFaction = tempPro
          bestLeft = left
          bestRight = right
        }
      }
    }
    (bestCutPoint, bestGini, bestDist, bestFaction, bestLeft, bestRight, missInsts)
  }

  def printTree(nodes: HashMap[Int, Node], root: Node, lev: Int) {
    val n = root
    val NTt = n.leaf
    val RTt = n.err
    val Rt = n.c
    //val alpha = (Rt - RTt) * 1.0 / (math.abs(NTt) - 1)
    val err = ",miss=" + n.miss
    val c = ",c=" + Rt
    val size = ",size=" + n.size
    println("->->" * lev + "i=" +
      root.i +
      //root.hit + ",mis=" +
      ",split=" + root.split
      + err + c + size)
    // println("-"*lev + node.toString)
    if (root.left > 0) printTree(nodes, nodes(root.left), lev + 1)
    if (root.right > 0) printTree(nodes, nodes(root.right), lev + 1)
  }
  def instanceFor(nodes: HashMap[Int, Node], lev: Int, x: Array[String], numIdx: HashSet[Int]): Double = {
    val n = nodes(lev)
    val i = n.i
    //    val t = if(i == -1){
    //      n.c
    //    }else{
    //      
    //    
    val isNumeric = if (numIdx.contains(i)) true else false
    //    
    val d = x(i).equalsIgnoreCase("?")
    //
    val m = if (d) {
      n.miss
    } else {
      //       
      //     
      val c = if (isNumeric) {
        //        
        val split = n.split.toDouble
        if (x(i).toDouble < split) {
          nodes(lev * 2).c
        } else {
          nodes(lev * 2 + 1).c
        }
        //        if (n.left > 0 && x(i).toDouble < split) {
        //          instanceFor(nodes, lev * 2, x, numIdx)
        //        } else if (n.right > 0 && x(i).toDouble >= split) {
        //          instanceFor(nodes, lev * 2 + 1, x, numIdx)
        //        } else {
        //          n.c
        //        }
        //
      } else {
        val split = n.split
        if (x(i).equalsIgnoreCase(split)) {
          nodes(lev * 2).c
        } else {
          nodes(lev * 2 + 1).c
        }
        //        if (n.left > 0 && x(i).equalsIgnoreCase(split)) {
        //          instanceFor(nodes, lev * 2, x, numIdx)
        //        } else if (n.right > 0 && !x(i).equalsIgnoreCase(split)) {
        //          instanceFor(nodes, lev * 2 + 1, x, numIdx)
        //        } else {
        //          n.c
        //        }
        //      }
        //      c
      }
      c
    }
    //    
    ////    }
    //    //println(t)
    m
  }

  def main(args: Array[String]): Unit = {
    var numIdx = new HashSet[Int]
    numIdx.+=(0)
    numIdx.+=(1)
    numIdx.+=(2)
    numIdx.+=(3)
    //    numIdx.+=(4)
    numIdx.+=(5)
    numIdx.+=(7)
    numIdx.+=(8)
    numIdx.+=(10)
    val insts = new RegInstances(numIdx)
    insts.read("E:/books/spark/ml/decisionTree/cpu.csv")

    val nodes = classifier(insts, 2)
    printTree(nodes, nodes(1), 0)
    insts.data.map(f => {
      println(instanceFor(nodes, 1, f.features, numIdx))
    })
  }

}