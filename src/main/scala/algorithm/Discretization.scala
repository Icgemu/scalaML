

package algorithm

import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import java.io.File
import cern.jet.stat.Gamma._
import scala.Array.canBuildFrom

//ÀëÉ¢»¯
object Discretization {
  case class LabeledFeature(label: String, features: Array[String])

  val SMALL = 1e-6
  val log2 = math.log(2)

  //var idx = HashMap[Int, HashSet[String]]()
  var data = ArrayBuffer[LabeledFeature]()
  // var numIdx = new HashSet[Int]
  //var root = Node(-1, null, "",0,0)
  var ite = 0
  def index(file: String) {
    var buff = Source.fromFile(new File(file))
    buff.getLines.toArray.map(line => {
      val arr = line.split(",")
      val label = arr.last.trim()
      val features = arr.slice(0, arr.length - 1)
      //      for (i <- 0 until features.length) {
      //        var v = features(i)
      //        //if("?".equalsIgnoreCase(v)){v="_NAN_"}
      //        //if (!idx.contains(i)) { idx.put(i, new HashSet[String]()) }
      //        //if (!idx(i).contains(v) && !numIdx.contains(i)) { idx(i) += v.trim() }
      //      }
      data.+=(LabeledFeature(label, features.map(f => f.trim())))
    })
    buff.close
  }
  def discretize(data: Array[LabeledFeature],
    indexArr: Array[Int]): HashMap[Int, Array[Double]] = {

    var map = new HashMap[Int, Array[Double]]
    for (i <- indexArr.toList) {
      val copy = data.sortBy(lf => { lf.features(i).toDouble })
      map(i) = subsets(copy, i)
    }
    map
  }

  def subsets(data: Array[LabeledFeature], attIndex: Int): Array[Double] = {
    var cutPoints = Array[Double]()

    val iClass = data.map(f => f.label).groupBy(f => f).keys.toArray

    val numClass = iClass.size
    val counts = Array.fill(2)(Array.fill(numClass)(0.0))
    var numInstances = 0.0
    data.map(lf => {
      numInstances += 1
      //weight
      counts(1)(labelToIndex(iClass, lf.label)) += 1
      //      counts(1)(labelToIndex(iClass, lf.label)) += lf.features(attIndex).toDouble
    })

    var priorCounts = Array.fill(numClass)(0.0)
    priorCounts = counts(1).map(f => f)

    // Entropy of the full set
    var priorEntropy = entropy(priorCounts)
    var bestEntropy = priorEntropy

    // Find best entropy.
    var bestCounts = Array.fill(2)(Array.fill(numClass)(0.0))
    var currentCutPoint = -Double.MaxValue
    var currentEntropy = 0.0
    var bestIndex = -1
    var numCutPoints = 0
    var bestCutPoint = -1.0

    for (i <- 0 until data.size - 1) {

      counts(0)(labelToIndex(iClass, data(i).label)) += 1
      counts(1)(labelToIndex(iClass, data(i).label)) -= 1

      //      counts(0)(labelToIndex(iClass, data(i).label)) += data(i).features(attIndex).toDouble
      //      counts(1)(labelToIndex(iClass, data(i).label)) -= data(i).features(attIndex).toDouble

      if (data(i).features(attIndex) < data(i + 1).features(attIndex)) {
        currentCutPoint = (data(i).features(attIndex).toDouble +
          data(i + 1).features(attIndex).toDouble) / 2.0;

        currentEntropy = entropyConditionedOnRows(counts)

        if (currentEntropy < bestEntropy) {
          bestCutPoint = currentCutPoint;
          bestEntropy = currentEntropy;
          bestIndex = i;
          bestCounts(0) = counts(0).map(f => f)
          bestCounts(1) = counts(1).map(f => f)
        }

        numCutPoints += 1;
      }
    }

    // Use worse encoding?
    //    if (!m_UseBetterEncoding) {
    //		numCutPoints = (lastPlusOne - first) - 1;
    numCutPoints = data.size - 1
    //    }

    // Checks if gain is zero
    val gain = priorEntropy - bestEntropy
    if (gain <= 0) {
      null
    } else {
      // Check if split is to be accepted
      //    if ((m_UseKononenko && KononenkosMDL(priorCounts, bestCounts,
      //					 numInstances, numCutPoints)) ||
      //	(!m_UseKononenko && FayyadAndIranisMDL(priorCounts, bestCounts,
      //					       numInstances, numCutPoints))) 
      //if (FayyadAndIranisMDL(priorCounts, bestCounts, numInstances, numCutPoints)) {
      if (KononenkosMDL(priorCounts, bestCounts, numInstances, numCutPoints)) {

        // Select split points for the left and right subsets
        var left: Array[Double] = null
        var right: Array[Double] = null
        val leftData = data.slice(0, bestIndex + 1)
        val rightData = data.slice(bestIndex + 1, data.size)
        if (leftData.size > 1) {
          left = subsets(leftData, attIndex)
        }
        if (rightData.size > 1) {
          right = subsets(rightData, attIndex)
        }

        // Merge cutpoints and return them
        if ((left == null) && (right) == null) {
          cutPoints = Array.fill(1)(bestCutPoint)
          //cutPoints(0) = bestCutPoint
        } else if (right == null) {
          cutPoints = Array.fill(left.length + 1)(0.0)
          left.copyToArray(cutPoints)
          cutPoints(left.length) = bestCutPoint
        } else if (left == null) {
          //cutPoints = new double[1 + right.length];
          cutPoints = Array.fill(1 + right.length)(0.0)
          cutPoints(0) = bestCutPoint
          //seq(0,1).map(f=>)
          var j = 1
          right.map(f => { cutPoints(j) = f; j += 1 })
          //right.copyToArray(xs, start, len)(right, 0, cutPoints, 1, right.length);
        } else {
          cutPoints = Array.fill(left.length + right.length + 1)(0.0)
          //System.arraycopy(left, 0, cutPoints, 0, left.length);
          var j = 0
          left.map(f => { cutPoints(j) = f; j += 1 })
          cutPoints(left.length) = bestCutPoint
          j += 1
          right.map(f => { cutPoints(j) = f; j += 1 })
          //System.arraycopy(right, 0, cutPoints, left.length + 1, right.length);
        }

        cutPoints
      } else
        null
    }
  }

  def lnFactorial(x: Double): Double = {
    logGamma(x + 1)
  }
  def log2Binomial(a: Double, b: Double): Double = {

    (lnFactorial(a) - lnFactorial(b) - lnFactorial(a - b)) / log2
  }
  def log2Multinomial(a: Double, bs: Array[Double]): Double = {

    var sum = 0.0

    for (i <- 0 until bs.length) {

      sum = sum + lnFactorial(bs(i));

    }
    (lnFactorial(a) - sum) / log2
  }

  def KononenkosMDL(priorCounts: Array[Double],
    bestCounts: Array[Array[Double]],
    numInstances: Double,
    numCutPoints: Int): Boolean = {

    var distPrior, instPrior = 0.0
    var distAfter, sum, instAfter = 0.0
    var before, after = 0.0
    var numClassesTotal = 0

    // Number of classes occuring in the set
    numClassesTotal = 0
    for (i <- 0 until priorCounts.length) {
      if (priorCounts(i) > 0) {
        numClassesTotal += 1
      }
    }

    // Encode distribution prior to split
    distPrior = log2Binomial(numInstances + numClassesTotal - 1,
      numClassesTotal - 1)

    // Encode instances prior to split.
    instPrior = log2Multinomial(numInstances, priorCounts)

    before = instPrior + distPrior

    // Encode distributions and instances after split.
    for (i <- 0 until bestCounts.length) {
      sum = bestCounts(i).sum
      distAfter += log2Binomial(sum + numClassesTotal - 1,
        numClassesTotal - 1)
      instAfter += log2Multinomial(sum,
        bestCounts(i))
    }

    // Coding cost after split
    after = math.log(numCutPoints) / log2 + distAfter + instAfter

    // Check if split is to be accepted
    (before > after)
  }

  def FayyadAndIranisMDL(
    priorCounts: Array[Double],
    bestCounts: Array[Array[Double]],
    numInstances: Double,
    numCutPoints: Int): Boolean = {

    var priorEntropy, entropy1, gain = 0.0
    var entropyLeft, entropyRight, delta = 0.0
    var numClassesTotal, numClassesRight, numClassesLeft = 0

    // Compute entropy before split.
    priorEntropy = entropy(priorCounts)

    // Compute entropy after split.
    entropy1 = entropyConditionedOnRows(bestCounts);

    // Compute information gain.
    gain = priorEntropy - entropy1;

    // Number of classes occuring in the set
    numClassesTotal = 0;
    for (i <- 0 until priorCounts.length) {
      if (priorCounts(i) > 0) {
        numClassesTotal += 1
      }
    }

    // Number of classes occuring in the left subset
    numClassesLeft = 0;
    for (i <- 0 until bestCounts(0).length) {
      if (bestCounts(0)(i) > 0) {
        numClassesLeft += 1
      }
    }

    // Number of classes occuring in the right subset
    numClassesRight = 0;
    for (i <- 0 until bestCounts(1).length) {
      if (bestCounts(1)(i) > 0) {
        numClassesRight += 1
      }
    }

    // Entropy of the left and the right subsets
    entropyLeft = entropy(bestCounts(0))

    entropyRight = entropy(bestCounts(1))

    // Compute terms for MDL formula
    delta = math.log(Math.pow(3, numClassesTotal) - 2) / log2 -
      ((numClassesTotal * priorEntropy) -
        (numClassesRight * entropyRight) -
        (numClassesLeft * entropyLeft))

    // Check if split is to be accepted
    (gain > (math.log(numCutPoints) / log2 + delta) / numInstances)
  }
  def lnFunc(num: Double): Double = {
    if (num <= 0) {
      0
    } else {
      num * Math.log(num)
    }
  }
  def eq(a: Double, b: Double): Boolean = {
    (a - b < SMALL) && (b - a < SMALL)
  }

  def entropy(array: Array[Double]): Double = {
    var returnValue = 0.0
    var sum = 0.0

    for (i <- 0 until array.length) {
      returnValue -= lnFunc(array(i))
      sum += array(i)
    }
    if (eq(sum, 0)) {
      0.0
    } else {
      (returnValue + lnFunc(sum)) / (sum * log2)
    }
  }

  def entropyConditionedOnRows(matrix: Array[Array[Double]]): Double = {

    var returnValue = 0.0
    var sumForRow, total = 0.0

    for (i <- 0 until matrix.length) {
      sumForRow = 0
      for (j <- 0 until matrix(i).length) {
        returnValue = returnValue + lnFunc(matrix(i)(j));
        sumForRow += matrix(i)(j)
      }
      returnValue = returnValue - lnFunc(sumForRow);
      total += sumForRow;
    }
    if (eq(total, 0)) {
      0.0
    } else {
      -returnValue / (total * log2)
    }
  }
  def labelToIndex(Arr: Array[String], label: String): Int = {
    var r = 0
    var j = -1
    val f = Arr.foreach(p => { if (p.equalsIgnoreCase(label)) { j = r }; r += 1 })
    j
  }

  def main(args: Array[String]): Unit = {
    //E:\books\spark\ml\discretization
    index("E:/books/spark/ml/discretization/iris2.csv")
    val r = discretize(data.toArray, Array(0, 1, 2, 3))
    println(r)
  }

}