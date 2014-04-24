package classifier.tree

import classifier.Model
import core.Instances
import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuffer
import core.Feature
import utils.Discretization
import scala.collection.mutable.HashSet

class DicisionStump(insts: Instances) extends TreeClassifierBase {

  def train(): DSModel = {
    val idx1 = insts.idxForNominal;
    val numIdx = insts.numIdx
    val data = insts.data
    val keyMaxSize = idx1.keys.max + 1
    var gini = Array.fill(keyMaxSize)(0.0)
    var split = Array.fill(keyMaxSize)("")
    var dist = Array.fill(keyMaxSize)(Array.fill(3)(HashMap[String, Double]()))
    var fration = Array.fill(keyMaxSize)(Array.fill(2)(0.0))
    var dataLeft = Array.fill(keyMaxSize)(ArrayBuffer[Feature]())
    var dataRight = Array.fill(keyMaxSize)(ArrayBuffer[Feature]())
    var dataMiss = Array.fill(keyMaxSize)(ArrayBuffer[Feature]())

    for (iAttr <- idx1.keys) {
      val isNumeric = numIdx.contains(iAttr)
      if (isNumeric) {

        var bestSplit = splitNumeric(1, data, iAttr)
        split(iAttr) = bestSplit._1 + ""
        gini(iAttr) = bestSplit._2
        fration(iAttr) = bestSplit._3
        dataLeft(iAttr) = bestSplit._4
        dataRight(iAttr) = bestSplit._5
        dataMiss(iAttr) = bestSplit._6
      } else {
        var bestSplit = splitNonimal(data, iAttr)
        split(iAttr) = bestSplit._1
        gini(iAttr) = bestSplit._2
        fration(iAttr) = bestSplit._3
        dataLeft(iAttr) = bestSplit._4
        dataRight(iAttr) = bestSplit._5
        dataMiss(iAttr) = bestSplit._6
      }
    }

    val bestAttr = maxIndex(gini)
    //val isNumeric = numIdx.contains(bestAttr)
    // val bestDist = dist(bestAttr)
    val bestSplit = split(bestAttr)
    var left = dataLeft(bestAttr)
    var right = dataRight(bestAttr)
    var miss = dataMiss(bestAttr)
    val fra = fration(bestAttr)

    val lc = valuefor(left)
    //val mc = valuefor(miss)
    val rc = valuefor(right)
    val ac = valuefor(data)
    //println("miss=>" + miss.size)
    //nodes(2) = new Node(index * 2, "", -1, -1, -1, index, lc, lc, 0.0, left.size, 1)
    //nodes(3) = new Node(index * 2 + 1, "", -1, -1, -1, index, rc, rc, 0.0, right.size, 1)

    println("split=>"+bestSplit)
    println("bestAttr=>"+bestAttr)
    println("fra=>"+fra.mkString(","))
    println("Array(lc, rc)=>"+Array(lc, rc).mkString(","))
    DSModel(DSNode(bestSplit, bestAttr, fra, Array(lc, rc)))
  }

  def valuefor(data: ArrayBuffer[Feature]): Map[String, Double] = {

    if (insts.isRegression) {
      val sum1 = data.map(f => f.target.toDouble * f.weight).sum
      val w = data.map(f => f.weight).sum
      Map("0" -> sum1 / w)
    } else {
      ratio(data)
    }
  }

  def labelfor(data: ArrayBuffer[Feature]): (String, Double, Double) = {

    val r = ratio(data);
    val sorted = r.toArray.sortBy(f => f._2).reverse
    val label = sorted(0)._1
    val (hit, mis) = hitAndMiss(data, label)
    (label, hit, mis)
  }
  def maxIndex(Arr: Array[Double]): Int = {
    var i = 0
    var max = Arr(0)
    var j = 0
    Arr.map(f => { if (f > max) { max = f; i = j }; j += 1 })
    i
  }

  def splitNonimal(instances: ArrayBuffer[Feature],
    iAttr: Int): (String, Double, Array[Double], ArrayBuffer[Feature], ArrayBuffer[Feature], ArrayBuffer[Feature]) = {
    var bestCutPoint = ""
    var bestGini = Double.MinValue
    var bestDist = Array.fill(2)(HashMap[String, Double]())
    var bestFaction = Array.fill(2)(0.0)
    val missInsts = instances.filter(p => p.features(iAttr).equalsIgnoreCase("?"))
    val hasValueInsts = instances.filter(p => { !p.features(iAttr).equalsIgnoreCase("?") })
    val allValue = hasValueInsts.map(f => f.features(iAttr)).toSet.toArray

    var bestLeft = ArrayBuffer[Feature]()
    var bestRight = ArrayBuffer[Feature]()
    var bestMiss = missInsts
    val parentDist = instances.groupBy(f => f.target).map(f => (f._1, f._2.map(f => f.weight).sum))

    for (currCutPoint <- allValue.toIterator) {
      val left = ArrayBuffer[Feature]()
      val right = ArrayBuffer[Feature]()
      val miss = missInsts.map(f => f)
      hasValueInsts.map(inst => {
        val f = inst.features(iAttr)
        if (!currCutPoint.equalsIgnoreCase(f)) {
          right += inst
        } else {
          left += inst
        }
      })
      val tempDist = Array.fill(2)(0.0)
      tempDist(0) = left.map(f => f.weight).sum * 1.0
      tempDist(1) = right.map(f => f.weight).sum * 1.0

      val tempSum = tempDist.sum
      val tempPro = tempDist.map(t => t * 1.0 / tempSum)

      if (!insts.isRegression) {
        for (i <- 0 until miss.size) {
          val t1 = miss(i).copy
          t1.weight = t1.weight * tempPro(0)
          left += t1
          val t2 = miss(i).copy
          t2.weight = t2.weight * tempPro(1)
          right += t2
        }
      }

      val sets = Map("0" -> left, "1" -> right, "2" -> miss)
      val currGini = computeGain(sets)
      //if(iAttr == 1)println(currCutPoint +","+currGini)
      if (currGini > bestGini && left.size > 2 && right.size > 2) {
        bestGini = currGini
        bestCutPoint = currCutPoint
        //bestDist = dist
        bestFaction = tempPro
        bestLeft = left
        bestRight = right
      }

    }
    println(iAttr +"=>"+(bestCutPoint, bestGini).toString+bestFaction(0)+","+bestFaction(1))
    (bestCutPoint, bestGini, bestFaction, bestLeft, bestRight, bestMiss)
  }

  def computeGain(
    all: Map[String, ArrayBuffer[Feature]]): Double = {

    if (insts.isRegression) {
      val left = all("0")
      val right = all("1")
      val miss = all("2")
      val avgleft = left.map(f => f.target.toDouble * f.weight).sum
      val avgright = right.map(f => f.target.toDouble * f.weight).sum
      val avgmiss = miss.map(f => f.target.toDouble * f.weight).sum

      val sqrleft = left.map(f => (f.target.toDouble) * (f.target.toDouble) * f.weight).sum
      val sqrright = right.map(f => (f.target.toDouble) * (f.target.toDouble) * f.weight).sum
      val sqrmiss = miss.map(f => (f.target.toDouble) * (f.target.toDouble) * f.weight).sum

      val swleft = left.map(f => f.weight).sum
      val swright = right.map(f => f.weight).sum
      val swmiss = miss.map(f => f.weight).sum

      val lossmiss = if(swmiss>0)sqrmiss - avgmiss * avgmiss / swmiss else 0.0
      val lossleft = if(swleft>0)sqrleft - avgleft * avgleft / swleft else 0.0
      val lossright =if(swright>0) sqrright - avgright * avgright / swright else 0.0

      -1.0 * (lossleft + lossright + lossmiss)
    } else {
      val p = (all("1") ++ all("0")).groupBy(f=>f.target).map(f=>(f._1,f._2.map(lf=>lf.weight).sum))
      val child = Array(
          all("0").groupBy(f=>f.target).map(f=>(f._1,f._2.map(lf=>lf.weight).sum)),
          all("1").groupBy(f=>f.target).map(f=>(f._1,f._2.map(lf=>lf.weight).sum)))
      //gain_entropy(all.filterNot(p=>p._1.equalsIgnoreCase("2")))
          computeGiniGain(child,p)
    }
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
  def computeGiniGain(childDist: Array[Map[String, Double]],
    parentDist: Map[String, Double]): Double = {
    val totalWeight = parentDist.values.sum
    //if (totalWeight==0) return 0;

    val leftWeight = childDist(0).values.sum
    val rightWeight = childDist(1).values.sum
    val parentGini = computeGini(parentDist, totalWeight)
    val leftGini = computeGini(childDist(0), leftWeight)
    val rightGini = computeGini(childDist(1), rightWeight)

    parentGini - leftWeight / totalWeight * leftGini - rightWeight / totalWeight * rightGini;
  }

  def splitNumeric(index: Int, instances: ArrayBuffer[Feature],
    iAttr: Int): (Double, Double, Array[Double], ArrayBuffer[Feature], ArrayBuffer[Feature], ArrayBuffer[Feature]) = {
    var bestCutPoint = -1.0
    var bestGini = Double.MinValue
    //var bestDist = Array.fill(3)(HashMap[String, Double]())
    var bestFaction = Array.fill(2)(0.0)
    val missInsts = instances.filter(p => p.features(iAttr).equalsIgnoreCase("?"))
    val hasValueInsts = instances.filter(p => { !p.features(iAttr).equalsIgnoreCase("?") })
    val allValue = hasValueInsts.map(f => f.features(iAttr).toDouble).toSet.toArray
    val sortedValue = allValue.sortBy(f => f)

    var bestLeft = ArrayBuffer[Feature]()
    var bestRight = ArrayBuffer[Feature]()
    var bestMiss = missInsts

    if (sortedValue.size < 2) {

    } else {
      for (i <- 0 until sortedValue.length - 1) {
        val currCutPoint = (sortedValue(i) + sortedValue(i + 1)) / 2.0
        val left = ArrayBuffer[Feature]()
        val right = ArrayBuffer[Feature]()
        val miss = missInsts.map(f => f)
        hasValueInsts.map(inst => {
          val f = inst.features(iAttr).toDouble
          if (f > currCutPoint) {
            right += inst
          } else {
            left += inst
          }
        })
        val tempDist = Array.fill(2)(0.0)
        tempDist(0) = left.map(f => f.weight).sum * 1.0
        tempDist(1) = right.map(f => f.weight).sum * 1.0

        val tempSum = tempDist.sum
        val tempPro = tempDist.map(t => t * 1.0 / tempSum)

        if (!insts.isRegression) {
          for (i <- 0 until miss.size) {
            val t1 = miss(i).copy
            t1.weight = t1.weight * tempPro(0)
            left += t1
            val t2 = miss(i).copy
            t2.weight = t2.weight * tempPro(1)
            right += t2
          }
        }

        val sets = Map("0" -> left, "1" -> right, "2" -> miss)
        val currGini = computeGain(sets)
        //if(iAttr == 1)println(currCutPoint +","+currGini)
        if (currGini > bestGini && left.size > 2 && right.size > 2) {
          bestGini = currGini
          bestCutPoint = currCutPoint
          //bestDist = dist
          bestFaction = tempPro
          bestLeft = left
          bestRight = right
        }
      }
    }
    println(iAttr +"=>"+(bestCutPoint, bestGini).toString+ bestFaction(0)+","+bestFaction(1))
    (bestCutPoint, bestGini, bestFaction, bestLeft, bestRight, bestMiss)
  }
}
case class DSNode(split: String, iAttr: Int, fraction: Array[Double], value: Array[Map[String, Double]])
case class DSModel(node: DSNode) extends Model {

  def getRegValue(lf: Feature, isNonimal: Boolean, i: Int, split: String): Double = {

    //val value = lf.target.toDouble
    //val s = split.toDouble
    val pre = if (lf.features(i).equalsIgnoreCase("?")) {
      node.fraction(0) * node.value(0)("0") + node.fraction(1) * node.value(1)("0")
    } else {
      val isTarget = if (isNonimal) lf.features(i).equalsIgnoreCase(split) else lf.features(i).toDouble <= split.toDouble
      if (isTarget) {
        node.value(0)("0")
      } else {
        node.value(1)("0")
      }
    }
    pre
  }
  
  def getLabelValue(lf: Feature, isNonimal: Boolean, i: Int, split: String): String = {

    val value = lf.target
    //val s = split.toDouble
    val pre = if (lf.features(i).equalsIgnoreCase("?")) {
      // val mapA = if(node.value(0).size>node.value(1).size)(0,1) else (1,0)

      val left = node.value(0).map(f => (f._1, f._2 * node.fraction(0)))
      val right = node.value(1).map(f => (f._1, f._2 * node.fraction(1)))
      val mapA = if (left.size >= right.size) (left, right) else (right, left)
      mapA._1.map(f => (f._1, f._2 + mapA._2.getOrElse(f._1, 0.0)))     
    } else {
      val isTarget = if (isNonimal) lf.features(i).equalsIgnoreCase(split) else lf.features(i).toDouble <= split.toDouble
      if (isTarget) {
        node.value(0)
      } else {
        node.value(1)
      }   
    }
    val t = pre.toArray.sortBy(f => f._2).reverse
    t(0)._1
  }
  def getLabelDist(lf: Feature, isNominal: Boolean, i: Int, split: String): Map[String,Double] = {

    val value = lf.target
    //val s = split.toDouble
    val pre = if (lf.features(i).equalsIgnoreCase("?")) {
      // val mapA = if(node.value(0).size>node.value(1).size)(0,1) else (1,0)

      val left = node.value(0).map(f => (f._1, f._2 * node.fraction(0)))
      val right = node.value(1).map(f => (f._1, f._2 * node.fraction(1)))
      val mapA = if (left.size >= right.size) (left, right) else (right, left)
      mapA._1.map(f => (f._1, f._2 + mapA._2.getOrElse(f._1, 0.0)))     
    } else {
      val isTarget = if (isNominal) lf.features(i).equalsIgnoreCase(split) else lf.features(i).toDouble <= split.toDouble
      if (isTarget) {
        node.value(0)
      } else {
        node.value(1)
      }   
    }
    pre
    //val t = pre.toArray.sortBy(f => f._2).reverse
    //t(0)._1
  }
  def predict(test: Instances): Double = {
    val i = node.iAttr
    val split = node.split
    var loss = 0.0

    val isReg = test.isRegression
    val isNonimal = !test.numIdx.contains(i)

    test.data.map(lf => {
      if (isReg) {
        val value = lf.target.toDouble
        val t = this.getRegValue(lf, isNonimal, i, split)
        loss += (value-t)*(value-t)
      } else {
        val t = this.getLabelValue(lf, isNonimal, i, split)
        if (t.equalsIgnoreCase(lf.target)) loss += 1.0
      }
    })
//    if((loss / test.data.size) <0.5){
//      print()
//    }
    loss / test.data.size
  }
}
object DicisionStump {

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
    
    val insts = new Instances(numIdx,false)
    insts.read("E:/books/spark/ml/decisionTree/labor.csv")
    //insts.all2Normal
    val (trainset, testset) = insts.stratify()

    val t = new DicisionStump(trainset)
    val model = t.train()

    val accu = model.predict(testset)
    println(accu);
  }
}