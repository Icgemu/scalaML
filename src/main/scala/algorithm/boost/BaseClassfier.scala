package algorithm.boost
import algorithm.tree.CARTv2._
import algorithm.Instances
import algorithm.LabeledFeature
import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuffer

object BaseClassfier {

  def M1(insts: Instances): (Int, Double, String, String, String, Double) = {

    val numIdx = insts.numIdx
    val idx1 = insts.idx
    val data = insts.data

    var iClass = data.groupBy(f => f.label).map(f => (f._1, f._2.size))
    val classSum = iClass.values.sum
    val classFraction = iClass.map(f => (f._1, f._2 * 1.0 / classSum))

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
    val ll = labelfor(left)
    val lr = labelfor(right)
    var err = (ll._3 + lr._3) / (ll._2 + ll._3 + lr._2 + lr._3)
    val alpha = math.log((1 - err) / err)
    (bestAttr, alpha, bestSplit, ll._1, lr._1, err)

  }

  def M2(insts: Instances,
    //Qtiy: Array[HashMap[String, Double]],
    Wi_notyi: Array[HashMap[String, Double]]): (Int, Double, String, Map[String, Double], Map[String, Double], Double) = {

    val numIdx = insts.numIdx
    val idx1 = insts.idx
    val data = insts.data

    var iClass = data.groupBy(f => f.label).map(f => (f._1, f._2.size))
    val classSum = iClass.values.sum
    val classFraction = iClass.map(f => (f._1, f._2 * 1.0 / classSum))

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

    var ll = labelfor2(left)
    var lr = labelfor2(right)
    //    val sum = ll.values.sum+lr.values.sum
    //    ll = ll.map(f=>(f._1,f._2/sum))
    //    lr = lr.map(f=>(f._1,f._2/sum))
    var err = 0.0

    for (i <- 0 until left.size) {
      val lf = left(i)
      val j = lf.i
      val f = lf.features(bestAttr)
      val label = lf.label

      //val N = Qtiy(j).size
      //val t = Qtiy(j).map(f=>f._2 * ll.getOrElse(f._1, 0.0)).sum
      //err += data(j).weight * (1 - ll(label) + t)   
      err += Wi_notyi(i).map(f => { f._2 * (1 - ll.getOrElse(label, 0.0) + ll.getOrElse(f._1, 0.0)) }).sum
      // }
    }
    for (i <- 0 until right.size) {
      val lf = right(i)
      val j = lf.i
      val f = lf.features(bestAttr)
      val label = lf.label

      //      val N = Qtiy(j).size
      //      val t = Qtiy(j).map(f=>f._2 * lr.getOrElse(f._1, 0.0)).sum
      //      err += data(j).weight * (1 - lr(label) + t)  
      err += Wi_notyi(i).map(f => { f._2 * (1 - lr.getOrElse(label, 0.0) + lr.getOrElse(f._1, 0.0)) }).sum
    }
    err = err * 0.5
    val beta_t = err / (1 - err) //0.5 * math.log((1 - err) / (err)) 

    //reweight
    for (i <- 0 until left.size) {
      val lf = left(i)
      val j = lf.i
      val f = lf.features(bestAttr)
      val label = lf.label
      Wi_notyi(j).+=(("_w_", lf.weight))
      Wi_notyi(j).map(t => {
        //print(Wi_notyi(j)(t._1))
        Wi_notyi(j)(t._1) = Wi_notyi(j)(t._1) *
          math.pow(beta_t, 0.5 * (1 - ll.getOrElse(t._1, 0.0) + ll.getOrElse(label, 0.0)))
        //math.exp(-1*beta_t*(1 + ll.getOrElse(t._1, 0.0) -ll(label)))
        //println("=>"+Wi_notyi(j)(t._1)) 
      })
      lf.weight = Wi_notyi(j).remove("_w_").get
    }
    for (i <- 0 until right.size) {
      val lf = data(i)
      val j = lf.i
      val f = lf.features(bestAttr)
      val label = lf.label
      Wi_notyi(j).+=(("_w_", lf.weight))
      Wi_notyi(j).map(t => {
        //print(Wi_notyi(j)(t._1))
        Wi_notyi(j)(t._1) = Wi_notyi(j)(t._1) *
          math.pow(beta_t, 0.5 * (1 - ll.getOrElse(t._1, 0.0) + ll.getOrElse(label, 0.0)))
        //math.exp(-1*beta_t*(1 + lr.getOrElse(t._1, 0.0) -lr(label)))
        //println("=>"+Wi_notyi(j)(t._1))
      })
      lf.weight = Wi_notyi(j).remove("_w_").get
    }
    //}

    //val alpha = math.log((1-err)/err)
    (bestAttr, beta_t, bestSplit, ll, lr, err)

  }

  def labelfor(data: ArrayBuffer[LabeledFeature]): (String, Double, Double) = {

    var iClass = data.groupBy(f => f.label).map(f => (f._1, f._2.map(t => t.weight).sum))
    val classSum = iClass.values.sum
    val classFraction = iClass.map(f => (f._1, f._2 * 1.0 / classSum))
    var max = -1.0
    var label = ""
    classFraction.map(c => { if (c._2 > max) { max = c._2; label = c._1 } })
    val hit = data.filter(d => d.label.equalsIgnoreCase(label)).map(f => { f.weight }).sum
    val mis = data.map(f => f.weight).sum - hit
    (label, hit, mis)
  }
  def labelfor2(data: ArrayBuffer[LabeledFeature]): Map[String, Double] = {

    //var iClass = data.groupBy(f => f.label).map(f => (f._1, f._2.map(t => t.weight).sum))
    var iClass = data.groupBy(f => f.label).map(f => (f._1, f._2.size))
    val classSum = iClass.values.sum
    val classFraction = iClass.map(f => (f._1, f._2 * 1.0 / classSum))
    //        var max = -1.0
    //        var label = ""
    //        classFraction.map(c => { if (c._2 > max) { max = c._2; label = c._1 } })
    //    val hit = data.filter(d => d.label.equalsIgnoreCase(label)).map(f=>{f.weight}).sum
    //    val mis = data.map(f=>f.weight).sum - hit
    //(label, hit, mis)
    //    val r = classFraction.map(f=>{
    //      if(f._1.equalsIgnoreCase(label))(f._1,1.0) else{(f._1,0.0)}
    //     })
    //     r
    classFraction
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

    val parentDist = instances.groupBy(f => f.label).map(f => (f._1, f._2.map(t => t.weight).sum))

    for (currCutPoint <- allValue.toIterator) {

      //val currCutPoint = (sortedValue(i) + sortedValue(i+1))/2.0
      var dist = Array.fill(2)(new HashMap[String, Double]())

      hasValueInsts.map(inst => {
        val f = inst.features(iAttr)
        if (currCutPoint.equalsIgnoreCase(f)) {
          val t = dist(1).getOrElse(inst.label, 0.0)
          dist(1)(inst.label) = t + inst.weight
        } else {
          dist(0)(inst.label) = dist(0).getOrElse(inst.label, 0.0) + inst.weight
        }
      })

      val tempDist = Array.fill(2)(0.0)
      tempDist(0) = dist(0).values.sum * 1.0
      tempDist(1) = dist(1).values.sum * 1.0

      val tempSum = tempDist.sum
      val tempPro = tempDist.map(t => t * 1.0 / tempSum)

      missInsts.map(inst => {
        for (i <- 0 until 2) {
          dist(i)(inst.label) = dist(i).getOrElse(inst.label, 0.0) + tempPro(i) * inst.weight
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

    val parentDist = instances.groupBy(f => f.label).map(f => (f._1, f._2.map(t => t.weight).sum))

    for (i <- 0 until sortedValue.length - 1) {
      val currCutPoint = (sortedValue(i) + sortedValue(i + 1)) / 2.0
      var dist = Array.fill(2)(HashMap[String, Double]())

      hasValueInsts.map(inst => {
        val f = inst.features(iAttr).toDouble
        if (f > currCutPoint) {
          dist(1)(inst.label) = dist(1).getOrElse(inst.label, 0.0) + inst.weight
        } else {
          dist(0)(inst.label) = dist(0).getOrElse(inst.label, 0.0) + inst.weight
        }
      })

      val tempDist = Array.fill(2)(0.0)
      tempDist(0) = dist(0).values.sum * 1.0
      tempDist(1) = dist(1).values.sum * 1.0

      val tempSum = tempDist.sum
      val tempPro = tempDist.map(t => t * 1.0 / tempSum)

      missInsts.map(inst => {
        for (i <- 0 until 2) {
          dist(i)(inst.label) = dist(i).getOrElse(inst.label, 0.0) + tempPro(i) * inst.weight
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

  def main(args: Array[String]): Unit = {}

}