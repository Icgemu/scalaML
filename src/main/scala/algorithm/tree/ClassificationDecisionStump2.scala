package algorithm.tree

import algorithm.RegInstances
import scala.collection.mutable.HashMap
import scala.collection.mutable.HashSet
import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import java.io.File
import algorithm.RegFeature
import algorithm.Instances
import algorithm.LabeledFeature
import algorithm.Discretization

object ClassificationDecisionStump2 {

  class Node(var index: Int, var split: String, var i: Int,
    var left: Int, var right: Int,
    var parent: Int,
    /**var err: Double,*/
    var c: Map[String, Double], var prob: Array[Double], var size: Int, var leaf: Int) {

  }

  var ite = 0
  val N = 1
  def classifier(insts: Instances, J: Int): HashMap[Int, Node] = {
    var nodes = HashMap[Int, Node]()
    val data = insts.data
    val idx = insts.idx
    val numIdx = insts.numIdx
    buildDT(nodes, data, idx, numIdx, 1)

    //get(nodes, J)
    //prune()
    //printTree(nodes(1), 0)
    //println("--------------")
    nodes
  }

  def buildDT(nodes: HashMap[Int, Node], data: ArrayBuffer[LabeledFeature],
    idx1: HashMap[Int, HashSet[String]],
    numIdx: HashSet[Int], index: Int) {

    var iClass = data.map(f => f.label).toSet
    //val classSum = iClass.values.sum
    //val classFraction = iClass.map(f => (f._1, f._2 * 1.0 / classSum))
    //val isLeaf = classFraction.filter(p => p._2 > 0.95).size > 0
    //val isLeaf = data.size == 1
    //println(data.size)
    //if (!isLeaf) {

    val keyMaxSize = idx1.keys.max + 1
    var gini = Array.fill(keyMaxSize)(0.0)
    var split = Array.fill(keyMaxSize)("")
    var dist = Array.fill(keyMaxSize)(Array.fill(3)(HashMap[String, Double]()))
    var fration = Array.fill(keyMaxSize)(Array.fill(2)(0.0))
    var dataLeft = Array.fill(keyMaxSize)(ArrayBuffer[LabeledFeature]())
    var dataRight = Array.fill(keyMaxSize)(ArrayBuffer[LabeledFeature]())
    var dataMiss = Array.fill(keyMaxSize)(ArrayBuffer[LabeledFeature]())
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

    println(index + "=>" + gini.mkString(","))
    //println("------------")
    val bestAttr = minIndex(gini)
    val isNumeric = numIdx.contains(bestAttr)
    val bestDist = dist(bestAttr)
    val bestSplit = split(bestAttr)
    val bestFration = fration(bestAttr)
    var left = dataLeft(bestAttr)
    var right = dataRight(bestAttr)
    var miss = dataMiss(bestAttr)

    //    val lc = valuefor(left)
    //    val mc = valuefor(miss)
    //    val rc = valuefor(right)
    //    val ac = valuefor(data)
    //    println("miss=>"+miss.size)
    iClass.map(f => {
      if (!bestDist(0).contains(f)) {
        bestDist(0)(f) = 0.0
      }
      if (!bestDist(1).contains(f)) {
        bestDist(1)(f) = 0.0
      }
    })
    val parentDist = data.groupBy(f => f.label).map(f => (f._1, f._2.map(f => f.weight).sum))
    nodes(index * 2) = new Node(index * 2, "", -1, -1, -1, index, bestDist(0).toMap, null, left.size, 1)
    nodes(index * 2 + 1) = new Node(index * 2 + 1, "", -1, -1, -1, index, bestDist(1).toMap, null, right.size, 1)
    nodes(index) = new Node(index, bestSplit, bestAttr, 2 * index, 2 * index + 1, -1, parentDist, bestFration, data.size, 2)

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
  def maxIndex(Arr: Array[Double]): Int = {
    var i = 0
    var max = Arr(0)
    var j = 0
    Arr.map(f => { if (f > max) { max = f; i = j }; j += 1 })
    i
  }
  def minIndex(Arr: Array[Double]): Int = {
    var i = 0
    var min = Arr(0)
    var j = 0
    Arr.map(f => { if (f < min) { min = f; i = j }; j += 1 })
    i
  }
  def splitNonimal(instances: ArrayBuffer[LabeledFeature],
    iAttr: Int): (String, Double, Array[HashMap[String, Double]], Array[Double], ArrayBuffer[LabeledFeature], ArrayBuffer[LabeledFeature], ArrayBuffer[LabeledFeature]) = {
    var bestCutPoint = ""
    var bestGini = Double.MaxValue
    var bestDist = Array.fill(2)(HashMap[String, Double]())
    var bestFaction = Array.fill(2)(0.0)
    val missInsts = instances.filter(p => p.features(iAttr).equalsIgnoreCase("?"))
    val hasValueInsts = instances.filter(p => { !p.features(iAttr).equalsIgnoreCase("?") })
    val allValue = hasValueInsts.map(f => f.features(iAttr)).toSet.toArray

    var bestLeft = ArrayBuffer[LabeledFeature]()
    var bestRight = ArrayBuffer[LabeledFeature]()
    var bestMiss = missInsts
    val parentDist = instances.groupBy(f => f.label).map(f => (f._1, f._2.map(f => f.weight).sum))

    for (currCutPoint <- allValue.toIterator) {

      //val currCutPoint = (sortedValue(i) + sortedValue(i+1))/2.0
      var dist = Array.fill(2)(HashMap[String, Double]())

      val left = ArrayBuffer[LabeledFeature]()
      val right = ArrayBuffer[LabeledFeature]()
      val miss = missInsts.map(f => f)
      hasValueInsts.map(inst => {
        val f = inst.features(iAttr)
        val label = inst.label
        if (!currCutPoint.equalsIgnoreCase(f)) {
          val t = dist(1).getOrElse(label, 0.0)
          dist(1)(label) = t + inst.weight
          right += inst
        } else {
          dist(0)(label) = dist(0).getOrElse(label, 0.0) + inst.weight
          left += inst
        }
      })
      val tempDist = Array.fill(2)(0.0)
      tempDist(0) = dist(0).values.sum * 1.0
      tempDist(1) = dist(1).values.sum * 1.0

      val tempSum = tempDist.sum
      val tempPro = tempDist.map(t => t * 1.0 / tempSum)

      //var w = miss.map(f=>f.weight).sum
      //val leftsize = (tempPro(0) * miss.size).toInt
      for (i <- 0 until miss.size) {
        //              val t = (math.random * miss.size).toInt
        //              val tt = miss.remove(t)
        val t1 = miss(i).copy
        t1.weight = t1.weight * tempPro(0)
        left += t1
        dist(0)(t1.label) = dist(0).getOrElse(t1.label, 0.0) + t1.weight
        val t2 = miss(i).copy
        t2.weight = t2.weight * tempPro(1)
        right += t2
        dist(1)(t2.label) = dist(1).getOrElse(t2.label, 0.0) + t2.weight
        //w -= tt.weight
      }
      //      miss.map(f => {
      //        val label = f.label
      //        dist(2)(label) = dist(2).getOrElse(label, 0.0) + f.weight
      //      })
      //Discretization.entropyConditionedOnRows(dist.toArray)
      val currGini = computeGiniGain(dist, parentDist)
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

  def computeGini(dist: Map[String, Double], total: Double): Double = {
    if (total == 0) {
      0.0
    } else {
      var v = 0.0
      dist.map(f => { v += (f._2 / total) * (f._2 / total) })
      1 - v
    }
  }
  def computeGiniGain(childDist: Array[HashMap[String, Double]],
    parentDist: Map[String, Double]): Double = {
    val t = childDist.map(f => f.values.toArray)
    Discretization.entropyConditionedOnRows(t)
    //    val totalWeight = parentDist.values.sum
    //    var parentGini = computeGini(parentDist, totalWeight)
    //    
    //    childDist.map(f => {
    //      val w = f.values.sum
    //      val gini = computeGini(f.toMap, w)
    //      parentGini = parentGini - (w/totalWeight)* gini
    //    })
    //    
    //    parentGini
  }

  def splitNumeric(index: Int, instances: ArrayBuffer[LabeledFeature],
    iAttr: Int): (Double, Double, Array[HashMap[String, Double]], Array[Double], ArrayBuffer[LabeledFeature], ArrayBuffer[LabeledFeature], ArrayBuffer[LabeledFeature]) = {
    var bestCutPoint = -1.0
    var bestGini = Double.MaxValue
    var bestDist = Array.fill(3)(HashMap[String, Double]())
    var bestFaction = Array.fill(2)(0.0)
    val missInsts = instances.filter(p => p.features(iAttr).equalsIgnoreCase("?"))
    val hasValueInsts = instances.filter(p => { !p.features(iAttr).equalsIgnoreCase("?") })
    val allValue = hasValueInsts.map(f => f.features(iAttr).toDouble).toSet.toArray
    val sortedValue = allValue.sortBy(f => f)

    var bestLeft = ArrayBuffer[LabeledFeature]()
    var bestRight = ArrayBuffer[LabeledFeature]()
    var bestMiss = missInsts
    //val parentDist = instances.groupBy(f => f.label).map(f => (f._1, f._2.size * 1.0))

    val parentDist = instances.groupBy(f => f.label).map(f => (f._1, f._2.map(f => f.weight).sum))

    if (sortedValue.size < 2) {

    } else {
      for (i <- 0 until sortedValue.length - 1) {
        if (index == 8) {
          print("")
        }
        val currCutPoint = (sortedValue(i) + sortedValue(i + 1)) / 2.0
        var dist = Array.fill(2)(HashMap[String, Double]())

        val left = ArrayBuffer[LabeledFeature]()
        val right = ArrayBuffer[LabeledFeature]()
        val miss = missInsts.map(f => f)

        hasValueInsts.map(inst => {
          val f = inst.features(iAttr).toDouble
          val label = inst.label
          if (f >= currCutPoint) {
            val t = dist(1).getOrElse(label, 0.0)
            dist(1)(label) = t + inst.weight
            right += inst
          } else {
            dist(0)(label) = dist(0).getOrElse(label, 0.0) + inst.weight
            left += inst
          }
        })
        val tempDist = Array.fill(2)(0.0)
        tempDist(0) = dist(0).values.sum * 1.0
        tempDist(1) = dist(1).values.sum * 1.0

        val tempSum = tempDist.sum
        val tempPro = tempDist.map(t => t * 1.0 / tempSum)

        //var w = miss.map(f=>f.weight).sum
        //val leftsize = (tempPro(0) * miss.size).toInt
        for (i <- 0 until miss.size) {
          //              val t = (math.random * miss.size).toInt
          //              val tt = miss.remove(t)
          val t1 = miss(i).copy
          t1.weight = t1.weight * tempPro(0)
          left += t1
          dist(0)(t1.label) = dist(0).getOrElse(t1.label, 0.0) + t1.weight
          val t2 = miss(i).copy
          t2.weight = t2.weight * tempPro(1)
          left += t2
          dist(1)(t2.label) = dist(1).getOrElse(t2.label, 0.0) + t2.weight
          //w -= tt.weight
        }
        //        miss.map(f => {
        //          val label = f.label
        //          dist(2)(label) = dist(2).getOrElse(label, 0.0) + f.weight
        //        })
        val currGini = computeGiniGain(dist, parentDist)

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
    (bestCutPoint, bestGini, bestDist, bestFaction, bestLeft, bestRight, bestMiss)
  }

  def printTree(nodes: HashMap[Int, Node], root: Node, lev: Int) {
    val n = root
    val NTt = n.leaf
    //val RTt = n.err
    //val Rt = n.
    //val alpha = (Rt - RTt) * 1.0 / (math.abs(NTt) - 1)
    //val err = ",miss=" + (if (n.miss != null) n.miss.mkString(",") else "")
    //val c = ",c=" + Rt.mkString(",")
    val size = ",size=" + n.size
    println("->->" * lev + "i=" +
      root.i + ", prob=" +
      n.c.mkString(",") +
      ",split=" + root.split
      + size)
    // println("-"*lev + node.toString)
    if (root.left > 0) printTree(nodes, nodes(root.left), lev + 1)
    if (root.right > 0) printTree(nodes, nodes(root.right), lev + 1)
  }
  def instanceFor(nodes: HashMap[Int, Node], lev: Int, x: Array[String], numIdx: HashSet[Int]): Map[String, Double] = {
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
      val p = n.prob
      val t1 = nodes(lev * 2).c
      val t2 = nodes(lev * 2 + 1).c
      val m = new HashMap[String, Double]()
      t1.map(f => { m(f._1) = m.getOrElse(f._1, 0.0) + f._2 * p(1) })
      t2.map(f => { m(f._1) = m.getOrElse(f._1, 0.0) + f._2 * p(1) })

      m.toMap
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
    val insts = new Instances(numIdx)
    insts.read("E:/books/spark/ml/decisionTree/labor.csv")

    val nodes = classifier(insts, 2)
    printTree(nodes, nodes(1), 0)
    insts.data.map(f => {
      println(instanceFor(nodes, 1, f.features, numIdx).mkString(","))
    })
  }

}