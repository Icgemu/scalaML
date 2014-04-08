package algorithm.tree

import algorithm.Instances
import scala.collection.mutable.HashMap
import scala.collection.mutable.HashSet
import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import java.io.File
import algorithm.LabeledFeature

object CARTv2 {

  def classifier(insts: Instances) :HashMap[Int, Node] = {
    val data = insts.data
    val idx = insts.idx
    val numIdx = insts.numIdx
    buildDT(data, idx, numIdx, 1)

    //get(J)
    //prune()
    nodes
  }
  
  case class Node(index: Int, split: String,
    left: Int, right: Int,
    parent: Int, err: Double,
    hit: Int, mis: Int, leaf: Int)

  var nodes = HashMap[Int, Node]()
  var ite = 0

  def prune() {
    val notLeaf = nodes.filter(p => p._2.leaf > 1)
    var minerr = Double.MaxValue
    var minIndex = -1
    var err = notLeaf.map(node => {

      val n = node._2
      val NTt = n.leaf
      val RTt = n.err
      val Rt = n.mis
      val alpha = (Rt - RTt) * 1.0 / (math.abs(NTt) - 1)
      if (alpha < minerr) { minerr = alpha; minIndex = node._1 }
    })

    if (minIndex % 2 == 0) { minIndex += 1 }
    truncateNode(nodes(1), minIndex)
    pruneNode(nodes(1))
  }

  def truncateNode(node: Node, maxIndex: Int) {
    if (node.index > maxIndex) { nodes.remove(node.index) }
    else {
      if (node.left > 0) truncateNode(nodes(node.left), maxIndex)
      if (node.right > 0) truncateNode(nodes(node.right), maxIndex)
    }
  }
  def pruneNode(node: Node) {
    if (nodes.contains(node.left)) { pruneNode(nodes(node.left)); pruneNode(nodes(node.right)) }
    else {
      nodes(node.index) = Node(node.index, node.split, -1, -1, node.parent,
        node.err, node.hit, node.mis, node.leaf)
    }
  }

  def buildDT(data: ArrayBuffer[LabeledFeature],
    idx1: HashMap[Int, HashSet[String]],
    numIdx: HashSet[Int], index: Int): Node = {

    var iClass = data.groupBy(f => f.label).map(f => (f._1, f._2.size))
    val classSum = iClass.values.sum
    val classFraction = iClass.map(f => (f._1, f._2 * 1.0 / classSum))
    val isLeaf = classFraction.filter(p => p._2 > 0.95).size > 0

    if (!isLeaf) {

      val keyMaxSize = idx1.keys.max + 1
      var gini = Array.fill(keyMaxSize)(0.0)
      var split = Array.fill(keyMaxSize)("")
      var dist = Array.fill(keyMaxSize)(Array.fill(2)(new HashMap[String, Double]))
      var fration = Array.fill(keyMaxSize)(Array.fill(2)(0.0))
      for (iAttr <- idx1.keys) {
        val isNumeric = numIdx.contains(iAttr)
        if (isNumeric) {
          var bestSplit = splitNumeric(data, iAttr, iClass.size)
          split(iAttr) = bestSplit._1 + ""
          gini(iAttr) = bestSplit._2
          dist(iAttr) = bestSplit._3
          fration(iAttr) = bestSplit._4
        } else {
          var bestSplit = splitNonimal(data, iAttr, iClass.size)
          split(iAttr) = bestSplit._1
          gini(iAttr) = bestSplit._2
          dist(iAttr) = bestSplit._3
          fration(iAttr) = bestSplit._4
        }
      }

      val bestAttr = maxIndex(gini)
      val isNumeric = numIdx.contains(bestAttr)
      val bestDist = dist(bestAttr)
      val bestSplit = split(bestAttr)
      var right = ArrayBuffer[LabeledFeature]()
      var left = ArrayBuffer[LabeledFeature]()
      val missValueData = data.filter(p => p.features(bestAttr).equalsIgnoreCase("?"))
      val valueData = data.filterNot(p => p.features(bestAttr).equalsIgnoreCase("?"))
      if (isNumeric) {
        val p = split(bestAttr).toDouble
        right = valueData.filter(lf => lf.features(bestAttr).toDouble > p)
        left = valueData.filter(lf => lf.features(bestAttr).toDouble <= p)

      } else {
        val p = split(bestAttr)
        right = valueData.filter(lf => { !lf.features(bestAttr).equalsIgnoreCase(p) })
        left = valueData.filter(lf => lf.features(bestAttr).equalsIgnoreCase(p))
      }
      //按比例分配缺失值实例
      if (missValueData.size > 0) {
        val leftpart = math.round(missValueData.size * fration(bestAttr)(0)).toInt
        val rightpart = missValueData.size - leftpart
        for (i <- 0 until leftpart) {
          val v = ((missValueData.size * math.random).toInt)
          left += missValueData(v)
          missValueData.remove(v)
        }
        right.++=(missValueData)
      }

      if (left.size > 2 && right.size > 2) {
        val leftNode = {
          var d = idx1.map(f => f)
          if (!isNumeric) d.remove(bestAttr)
          val node = buildDT(left, d, numIdx, 2 * index)
          nodes(2 * index) = node
          node
        }
        val rightNode = {
          var d = idx1.map(f => f)
          if (!isNumeric) { d(bestAttr).remove(bestSplit) }
          val node = buildDT(right, d, numIdx, 2 * index + 1)
          nodes(2 * index + 1) = node
          node
        }
        val parent = if (index % 2 == 0) { index / 2 } else { (index - 1) / 2 }
        val NTt = leftNode.leaf + rightNode.leaf

        val l = labelfor(data)
        val node = Node(index,
          split(bestAttr),
          2 * index, 2 * index + 1, parent,
          leftNode.err + rightNode.err,
          l._2,
          l._3,
          NTt)
        nodes(index) = node
        node
      } else {
        val parent = if (index % 2 == 0) { index / 2 } else { (index - 1) / 2 }
        val l = labelfor(data)
        val node = Node(index, l._1, -1, -1, parent, l._3, l._2, l._3, 1)
        nodes(index) = node
        node
      }

    } else {
      val parent = if (index % 2 == 0) { index / 2 } else { (index - 1) / 2 }
      val l = labelfor(data)
      val node = Node(index, l._1, -1, -1, parent, l._3, l._2, l._3, 1)
      nodes(index) = node
      node
    }
  }
  def labelfor(data: ArrayBuffer[LabeledFeature]): (String, Int, Int) = {

    var iClass = data.groupBy(f => f.label).map(f => (f._1, f._2.size))
    val classSum = iClass.values.sum
    val classFraction = iClass.map(f => (f._1, f._2 * 1.0 / classSum))
    var max = -1.0
    var label = ""
    classFraction.map(c => { if (c._2 > max) { max = c._2; label = c._1 } })
    val hit = data.filter(d => d.label.equalsIgnoreCase(label)).size
    val mis = data.size - hit
    (label, hit, mis)
  }
  def maxIndex(Arr: Array[Double]): Int = {
    var i = 0
    var max = Arr(0)
    var j = 0
    Arr.map(f => { if (f > max) { max = f; i = j }; j += 1 })
    i
  }
  def splitNonimal(instances: ArrayBuffer[LabeledFeature],
    iAttr: Int,
    numClass: Int): (String, Double, Array[HashMap[String, Double]], Array[Double]) = {
    var bestCutPoint = ""
    var bestGini = -Double.MaxValue
    var bestDist = Array.fill(2)(HashMap[String, Double]())
    var bestFaction = Array.fill(2)(0.0)
    val missInsts = instances.filter(p => p.features(iAttr).equalsIgnoreCase("?"))
    val hasValueInsts = instances.filter(p => { !p.features(iAttr).equalsIgnoreCase("?") })
    val allValue = hasValueInsts.map(f => f.features(iAttr)).toSet.toArray

    val parentDist = instances.groupBy(f => f.label).map(f => (f._1, f._2.size * 1.0))

    for (currCutPoint <- allValue.toIterator) {

      //val currCutPoint = (sortedValue(i) + sortedValue(i+1))/2.0
      var dist = Array.fill(2)(new HashMap[String, Double]())

      hasValueInsts.map(inst => {
        val f = inst.features(iAttr)
        if (currCutPoint.equalsIgnoreCase(f)) {
          val t = dist(1).getOrElse(inst.label, 0.0)
          dist(1)(inst.label) = t + 1.0
        } else {
          dist(0)(inst.label) = dist(0).getOrElse(inst.label, 0.0) + 1
        }
      })

      val tempDist = Array.fill(2)(0.0)
      tempDist(0) = dist(0).values.sum * 1.0
      tempDist(1) = dist(1).values.sum * 1.0

      val tempSum = tempDist.sum
      val tempPro = tempDist.map(t => t * 1.0 / tempSum)

      missInsts.map(inst => {
        for (i <- 0 until 2) {
          dist(i)(inst.label) = dist(i).getOrElse(inst.label, 0.0) + tempPro(i) * 1
        }
      })
      val currGini = computeGiniGain(dist, parentDist)
      if (currGini > bestGini) {
        bestGini = currGini
        bestCutPoint = currCutPoint
        bestDist = dist
        bestFaction = tempPro
      }

    }
    (bestCutPoint, bestGini, bestDist, bestFaction)
  }
  def splitNumeric(instances: ArrayBuffer[LabeledFeature],
    iAttr: Int,
    numClass: Int): (Double, Double, Array[HashMap[String, Double]], Array[Double]) = {
    var bestCutPoint = -1.0
    var bestGini = -Double.MaxValue
    var bestDist = Array.fill(2)(HashMap[String, Double]())
    var bestFaction = Array.fill(2)(0.0)
    val missInsts = instances.filter(p => p.features(iAttr).equalsIgnoreCase("?"))
    val hasValueInsts = instances.filter(p => { !p.features(iAttr).equalsIgnoreCase("?") })
    val allValue = hasValueInsts.map(f => f.features(iAttr).toDouble).toSet.toArray
    val sortedValue = allValue.sortBy(f => f)

    val parentDist = instances.groupBy(f => f.label).map(f => (f._1, f._2.size * 1.0))

    for (i <- 0 until sortedValue.length - 1) {
      val currCutPoint = (sortedValue(i) + sortedValue(i + 1)) / 2.0
      var dist = Array.fill(2)(HashMap[String, Double]())

      hasValueInsts.map(inst => {
        val f = inst.features(iAttr).toDouble
        if (f > currCutPoint) {
          dist(1)(inst.label) = dist(1).getOrElse(inst.label, 0.0) + 1
        } else {
          dist(0)(inst.label) = dist(0).getOrElse(inst.label, 0.0) + 1
        }
      })

      val tempDist = Array.fill(2)(0.0)
      tempDist(0) = dist(0).values.sum * 1.0
      tempDist(1) = dist(1).values.sum * 1.0

      val tempSum = tempDist.sum
      val tempPro = tempDist.map(t => t * 1.0 / tempSum)

      missInsts.map(inst => {
        for (i <- 0 until 2) {
          dist(i)(inst.label) = dist(i).getOrElse(inst.label, 0.0) + tempPro(i) * 1
        }
      })
      val currGini = computeGiniGain(dist, parentDist)
      if (currGini > bestGini) {
        bestGini = currGini
        bestCutPoint = currCutPoint
        bestDist = dist
        bestFaction = tempPro
      }
    }
    (bestCutPoint, bestGini, bestDist, bestFaction)
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
    val totalWeight = parentDist.values.sum
    //if (totalWeight==0) return 0;

    val leftWeight = childDist(0).values.sum
    val rightWeight = childDist(1).values.sum

    val parentGini = computeGini(parentDist, totalWeight)
    val leftGini = computeGini(childDist(0).toMap, leftWeight)
    val rightGini = computeGini(childDist(1).toMap, rightWeight)

    parentGini - leftWeight / totalWeight * leftGini - rightWeight / totalWeight * rightGini;
  }

  def log2(a: Double): Double = {
    math.log(a) / math.log(2)
  }
  def printTree(root: Node, lev: Int) {
    val n = root
    val NTt = n.leaf
    val RTt = n.err
    val Rt = n.mis
    val alpha = (Rt - RTt) * 1.0 / (math.abs(NTt) - 1)
    val err = if (NTt > 1) { ",err=" + alpha } else { "" }
    println("->->" * lev + "i=" +
      root.index + ",hit=" +
      root.hit + ",mis=" +
      root.mis + ",split=" + root.split
      + err)
    // println("-"*lev + node.toString)
    if (root.left > 0) printTree(nodes(root.left), lev + 1)
    if (root.right > 0) printTree(nodes(root.right), lev + 1)
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

    classifier(insts)
    printTree(nodes(1), 0)
  }

}