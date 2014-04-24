package regression.tree

import regression.Regression
import regression.Model
import core.Instances
import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuffer
import core.Feature
import scala.collection.mutable.HashSet

class CARTRegression(insts: Instances, J: Int) extends Regression {
  var nodes = HashMap[Int, Node]()
  //val data = insts.data
  //val idx = insts.idxForNominal
  val numIdx = insts.numIdx

  def train(): CARTRegModel = {
    val data = insts.data
    val idx = insts.idxForNominal
    train0(data,idx,1)
    prune()
    CARTRegModel(nodes)
  }

  def prune() {
    val set = HashSet(2, 3)
    val j = if (J % 2 == 0) J else J + 1
    val okset = HashSet[Int]()
    while ((set.size + okset.size) < j && set.size > 0) {
      val sl = set.toArray.sortBy(f => f).head
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
    val l = nodes(okset.toArray.sortBy(f => f).last).index
    truncateNode(nodes(1), l)
    pruneNode(nodes(1))
    val remove = nodes.filter(p => {
      p._1 > l
    })
    remove.map(f => {
      nodes.remove(f._1)
    })
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
      nodes(node.index) = new Node(node.index, node.split, node.i, -1, -1, node.parent,
        node.err, node.c, node.size, node.leaf)
    }
  }
  def train0(data: ArrayBuffer[Feature],
    idx1: HashMap[Int, HashSet[String]], index: Int): Node = {

    val isLeaf = data.size == 1
    if (!isLeaf) {

      val keyMaxSize = idx1.keys.max + 1
      var gini = Array.fill(keyMaxSize)(0.0)
      var split = Array.fill(keyMaxSize)("")
      var dist = Array.fill(keyMaxSize)(Array.fill(2)(0.0))
      var fration = Array.fill(keyMaxSize)(Array.fill(2)(0.0))
      var dataLeft = Array.fill(keyMaxSize)(ArrayBuffer[Feature]())
      var dataRight = Array.fill(keyMaxSize)(ArrayBuffer[Feature]())
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
        } else {
          var bestSplit = splitNonimal(data, iAttr)
          split(iAttr) = bestSplit._1
          gini(iAttr) = bestSplit._2
          dist(iAttr) = bestSplit._3
          fration(iAttr) = bestSplit._4
          dataLeft(iAttr) = bestSplit._5
          dataRight(iAttr) = bestSplit._6
        }
      }
      val bestAttr = minIndex(gini)
      val isNumeric = numIdx.contains(bestAttr)
      val bestDist = dist(bestAttr)
      val bestSplit = split(bestAttr)
      var left = dataLeft(bestAttr)
      var right = dataRight(bestAttr)

      if (left.size > 2 && right.size > 2) {
        val leftNode = {
          var d = idx1.map(f => f)
          if (!isNumeric) d(bestAttr).remove(bestSplit)
          val node = train0(left, d, 2 * index)
          nodes(2 * index) = node
          node
        }
        val rightNode = {
          var d = idx1.map(f => f)
          if (!isNumeric) { d(bestAttr).remove(bestSplit) }
          val node = train0(right, d, 2 * index + 1)
          nodes(2 * index + 1) = node
          node
        }
        val parent = if (index % 2 == 0) { index / 2 } else { (index - 1) / 2 }
        val NTt = leftNode.leaf + rightNode.leaf

        val l = valuefor(data)
        val node = new Node(index,
          split(bestAttr),
          bestAttr,
          2 * index, 2 * index + 1, parent,
          leftNode.err + rightNode.err,
          l, data.size,
          NTt)
        nodes(index) = node
        node
      } else {
        val parent = if (index % 2 == 0) { index / 2 } else { (index - 1) / 2 }
        val l = valuefor(data)
        val node = new Node(index, "", -1, -1, -1, parent, l, l, data.size, 1)
        nodes(index) = node
        node
      }

    } else {
      val parent = if (index % 2 == 0) { index / 2 } else { (index - 1) / 2 }
      val l = valuefor(data)
      val node = new Node(index, "", -1, -1, -1, parent, l, l, data.size, 1)
      nodes(index) = node
      node
    }

  }
  def valuefor(data: ArrayBuffer[Feature]): Double = {
    val sum1 = data.map(f => f.target.toDouble).sum
    sum1 / data.size

  }
  def minIndex(Arr: Array[Double]): Int = {
    var i = 0
    var min = Arr(0)
    var j = 0
    Arr.map(f => { if (f < min) { min = f; i = j }; j += 1 })
    i
  }
  def splitNonimal(instances: ArrayBuffer[Feature],
    iAttr: Int): (String, Double, Array[Double], Array[Double], ArrayBuffer[Feature], ArrayBuffer[Feature]) = {
    var bestCutPoint = ""
    var bestGini = Double.MaxValue
    var bestDist = Array.fill(2)(0.0)
    var bestFaction = Array.fill(2)(0.0)
    val missInsts = instances.filter(p => p.features(iAttr).equalsIgnoreCase("?"))
    val hasValueInsts = instances.filter(p => { !p.features(iAttr).equalsIgnoreCase("?") })
    val allValue = hasValueInsts.map(f => f.features(iAttr)).toSet.toArray

    var bestLeft = ArrayBuffer[Feature]()
    var bestRight = ArrayBuffer[Feature]()
    for (currCutPoint <- allValue.toIterator) {

      var dist = Array.fill(2)(0.0)
      val left = ArrayBuffer[Feature]()
      val right = ArrayBuffer[Feature]()
      val miss = missInsts.map(f => f)
      hasValueInsts.map(inst => {
        val f = inst.features(iAttr)
        if (currCutPoint.equalsIgnoreCase(f)) {
          val t = dist(1)
          dist(1) = t + 1
          right += inst
        } else {
          dist(0) = dist(0) + 1
          left += inst
        }
      })

      val tempSum = dist.sum
      val tempPro = dist.map(t => t * 1.0 / tempSum)

      val leftsize = (tempPro(0) * miss.size).toInt
      for (i <- 0 until leftsize) {
        val t = (math.random * miss.size).toInt
        left += miss.remove(t)
      }
      right ++= miss

      val avgleft = left.map(f => f.target.toDouble).sum / left.size
      val avgright = right.map(f => f.target.toDouble).sum / right.size
      val lossleft = left.map(f => (f.target.toDouble - avgleft) * (f.target.toDouble - avgleft)).sum
      val lossright = right.map(f => (f.target.toDouble - avgright) * (f.target.toDouble - avgright)).sum
      val currGini = lossleft + lossright //computeGiniGain(dist, parentDist)
      if (currGini < bestGini && left.size > 2 && right.size > 2) {
        bestGini = currGini
        bestCutPoint = currCutPoint
        bestDist = dist
        bestFaction = tempPro
        bestLeft = left
        bestRight = right
      }

    }
    (bestCutPoint, bestGini, bestDist, bestFaction, bestLeft, bestRight)
  }
  def splitNumeric(index: Int, instances: ArrayBuffer[Feature],
    iAttr: Int): (Double, Double, Array[Double], Array[Double], ArrayBuffer[Feature], ArrayBuffer[Feature]) = {
    var bestCutPoint = -1.0
    var bestGini = Double.MaxValue
    var bestDist = Array.fill(2)(0.0)
    var bestFaction = Array.fill(2)(0.0)
    val missInsts = instances.filter(p => p.features(iAttr).equalsIgnoreCase("?"))
    val hasValueInsts = instances.filter(p => { !p.features(iAttr).equalsIgnoreCase("?") })
    val allValue = hasValueInsts.map(f => f.features(iAttr).toDouble).toSet.toArray
    val sortedValue = allValue.sortBy(f => f)

    var bestLeft = ArrayBuffer[Feature]()
    var bestRight = ArrayBuffer[Feature]()
    val parentReg = instances.map(f => f.target.toDouble).sum / instances.size

    if (sortedValue.size < 2) {

    } else {
      for (i <- 0 until sortedValue.length - 1) {
        if (index == 8) {
          print("")
        }
        val currCutPoint = (sortedValue(i) + sortedValue(i + 1)) / 2.0
        var dist = Array.fill(2)(0.0)

        val left = ArrayBuffer[Feature]()
        val right = ArrayBuffer[Feature]()
        val miss = missInsts.map(f => f)
        hasValueInsts.map(inst => {
          val f = inst.features(iAttr).toDouble
          if (f > currCutPoint) {
            dist(1) = dist(1) + 1
            right += inst
          } else {
            dist(0) = dist(0) + 1
            left += inst
          }
        })

        val tempDist = Array.fill(2)(0.0)
        val tempSum = dist.sum
        val tempPro = dist.map(t => t * 1.0 / tempSum)

        val leftsize = (tempPro(0) * miss.size).toInt
        for (i <- 0 until leftsize) {
          val t = (math.random * miss.size).toInt
          left += miss.remove(t)
        }
        right ++= miss

        val avgleft = left.map(f => f.target.toDouble).sum / left.size
        val avgright = right.map(f => f.target.toDouble).sum / right.size
        val lossleft = left.map(f => (f.target.toDouble - avgleft) * (f.target.toDouble - avgleft)).sum
        val lossright = right.map(f => (f.target.toDouble - avgright) * (f.target.toDouble - avgright)).sum
        val currGini = lossleft + lossright //computeGiniGain(dist, parentDist)

        if (currGini < bestGini && left.size > 2 && right.size > 2) {
          bestGini = currGini
          bestCutPoint = currCutPoint
          bestDist = dist
          bestFaction = tempPro
          bestLeft = left
          bestRight = right
        }
      }
    }
    (bestCutPoint, bestGini, bestDist, bestFaction, bestLeft, bestRight)
  }

 
}
 case class CARTRegModel(nodes:HashMap[Int, Node]) extends Model {

    def getRegValue(feature: Feature,numIdx: HashSet[Int], lev: Int): Double = {
      
      val x = feature.features
      val n = nodes(lev)
      val i = n.i
      val t = if (i == -1) {
        n.c
      } else {
        val isNumeric = if (numIdx.contains(i)) true else false
        val d = x(i).equalsIgnoreCase("?")
        val m = if (d) {
          0.0
        } else {
          val c = if (isNumeric) {
            val split = n.split.toDouble
            if (n.left > 0 && x(i).toDouble < split) {
              getRegValue(feature, numIdx,lev * 2)
            } else if (n.right > 0 && x(i).toDouble >= split) {
              getRegValue(feature, numIdx,lev * 2 + 1)
            } else {
              n.c
            }
          } else {
            val split = n.split
            if (n.left > 0 && x(i).equalsIgnoreCase(split)) {
              getRegValue(feature, numIdx,lev * 2)
            } else if (n.right > 0 && !x(i).equalsIgnoreCase(split)) {
              getRegValue(feature, numIdx,lev * 2 + 1)
            } else {
              n.c
            }
          }
          c
        }
        m
      }
      t
    }
    def predict(test: Instances): Double = {
      var r = 0.0
      test.data.foreach(f => {
        val c = getRegValue(f, test.numIdx,1)
        r += (f.target.toDouble - c) * (f.target.toDouble - c)
      })
      r / test.data.size
    }
  }

class Node(var index: Int, var split: String, var i: Int,
  var left: Int, var right: Int,
  var parent: Int, var err: Double,
  var c: Double, var size: Int, var leaf: Int) {
}

object CARTRegression {

  def main(args: Array[String]): Unit = {

    var numIdx = new HashSet[Int]
    numIdx.+=(0)
    numIdx.+=(1)
    numIdx.+=(2)
    numIdx.+=(3)
    numIdx.+=(4)
    numIdx.+=(5)
//        numIdx.+=(7)
//        numIdx.+=(8)
//        numIdx.+=(10)

    val insts = new Instances(numIdx)
    insts.read("E:/books/spark/ml/decisionTree/cpu.csv")
    //    insts.all2Normal
    val (trainset, testset) = insts.stratify()

    val t = new CARTRegression(trainset, 12)
    val model = t.train()

    val loss = model.predict(testset)
    println(loss);
  }
}