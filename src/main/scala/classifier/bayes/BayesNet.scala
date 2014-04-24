package classifier.bayes

import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashSet
import scala.collection.mutable.HashMap
import classifier.Classifier
import core.Instances
import classifier.Model
import core.Feature

class BayesNet(insts: Instances) extends Classifier {

  case class DiscreteEstimatorBayes(m_Counts: Array[Double], m_SumOfCounts: Double, m_nSymbols: Int, m_fPrior: Double)
  case class Node(i: Int, par: ArrayBuffer[Node], cpt: Array[Double], m_Distributions: Array[DiscreteEstimatorBayes])
  case class LabeledFeature(label: String, features: Array[String])
  var idx = HashMap[Int, HashSet[String]]()
  var data = ArrayBuffer[LabeledFeature]()
  var nodes = Array[Node]()

  val m_fAlpha = 0.5
  val maxParents = 2
  def train(): Model = {
    insts.data.map(f => {
      val label = f.target
      var features = (ArrayBuffer(label) ++ f.features).toArray //.slice(0, arr.length - 1)
      for (i <- 0 until features.length) {
        var v = features(i)
        if (!idx.contains(i)) { idx.put(i, new HashSet[String]()) }
        if (!idx(i).contains(v)) { idx(i) += v.trim() }
      }
      data.+=(LabeledFeature(label, features.map(f => f.trim())))
    })

    initStructure()
    buildStructure()
    estimateCPTs()

    BNetModel()
  }

  def initStructure() {
    nodes = Array.fill(idx.size)(Node(-1, null, null, null))
    val all = idx.values.map(f => f.size).sum //cache all node
    idx.map(node => {
      nodes(node._1) = Node(node._1,
        new ArrayBuffer[Node](),
        Array.fill(idx(node._1).size * all)(0.0), null)
    })
  }

  def buildStructure() {

    var fBaseScores = Array.fill(nodes.size)(0.0)

    fBaseScores = nodes.map(node => {
      val attr = node.i
      calcNodeScore(attr)
    })

    // K2 algorithm: greedy search restricted by ordering 
    for (iOrder <- 1 until idx.size) {
      val iAttribute = nodes(iOrder).i
      var fBestScore = fBaseScores(iAttribute)

      var bProgress = (nodes(iAttribute).par.size < maxParents)
      while (bProgress) {
        var nBestAttribute = -1;
        for (iOrder2 <- 0 until iOrder) {
          val iAttribute2 = nodes(iOrder2).i
          val fScore = calcScoreWithExtraParent(iAttribute, iAttribute2)
          if (fScore > fBestScore) {
            fBestScore = fScore
            nBestAttribute = iAttribute2
          }
        }
        if (nBestAttribute != -1) {
          nodes(iAttribute).par.+=(nodes(nBestAttribute))

          fBaseScores(iAttribute) = fBestScore

          bProgress = (nodes(iAttribute).par.size < maxParents)
        } else {
          bProgress = false
        }
      }
    }

  }

  def estimateCPTs() {
    initCPTs
    data.map(lf => {
      for (iAttribute <- 0 until idx.size) {
        var iCPT = 0
        val index = indexInSet(lf.features(iAttribute), idx(iAttribute))

        for (iParent <- 0 until nodes(iAttribute).par.size) {
          val nParent = nodes(iAttribute).par(iParent).i
          iCPT = iCPT * idx(nParent).size + indexInSet(lf.features(nParent), idx(nParent))
        }
        val dist = nodes(iAttribute).m_Distributions(iCPT)
        var m_Counts = dist.m_Counts
        var m_SumOfCounts = dist.m_SumOfCounts
        m_Counts(index) += 1
        m_SumOfCounts += 1
        nodes(iAttribute).m_Distributions(iCPT) =
          DiscreteEstimatorBayes(m_Counts, m_SumOfCounts, dist.m_nSymbols, dist.m_fPrior)
      }
    })
  }

  def initCPTs() {
    // Reserve space for CPTs
    var nMaxParentCardinality = 1
    for (iAttribute <- 0 until idx.size) {
      var n = 1
      nodes(iAttribute).par.map(f => { n = n * idx(f.i).size })
      if (n > nMaxParentCardinality) {
        nMaxParentCardinality = n
      }
    }

    // Reserve plenty of memory
    //m_Distributions = new Estimator[instances.numAttributes()][nMaxParentCardinality];

    // estimate CPTs
    for (iAttribute <- 0 until idx.size) {
      var n = 1
      nodes(iAttribute).par.map(f => { n = n * idx(f.i).size })
      var dist = nodes(iAttribute).m_Distributions
      if (dist == null) {
        dist = Array.fill(nMaxParentCardinality)(null)
      }
      val node = nodes(iAttribute)
      nodes(iAttribute) = Node(node.i, node.par, node.cpt, dist)
      for (iParent <- 0 until n) {
        val m_fPrior = m_fAlpha;
        val m_nSymbols = idx(iAttribute).size;
        val m_Counts = Array.fill(idx(iAttribute).size)(m_fPrior)

        val m_SumOfCounts = m_fPrior * m_nSymbols;
        nodes(iAttribute).m_Distributions(iParent) =
          DiscreteEstimatorBayes(m_Counts, m_SumOfCounts, m_nSymbols, m_fPrior)
      }
    }
  }

  def calcNodeScore(attr: Int): Double = {
    var numValues = idx(attr).size
    var parArr = nodes(attr).par
    var nCardinality = if (parArr.size < 1) 1 else {
      var c = 1
      parArr.map(node => c = idx(node.i).size * c)
      c
    }
    var nCounts = Array.fill(numValues * nCardinality)(0)
    data.map(lf => {
      val fs = lf.features(attr)
      var icpt = 0
      for (par <- parArr) {
        icpt = icpt * idx(par.i).size + indexInSet(lf.features(par.i), idx(par.i))
      }
      nCounts(numValues * icpt + indexInSet(fs, idx(attr))) += 1
    })
    calcScoreOfCounts(nCounts, nCardinality, numValues)
  }

  def calcScoreOfCounts(nCounts: Array[Int],
    nCardinality: Int,
    numValues: Int): Double = {
    var fLogScore = 0.0

    for (iParent <- 0 until nCardinality) {
      //BAYES
      //      var nSumOfCounts = 0.0
      //
      //      for (iSymbol <- 0 until numValues) {
      //        if (m_fAlpha + nCounts(iParent * numValues + iSymbol) != 0) {
      //          fLogScore += logGamma(m_fAlpha + nCounts(iParent * numValues + iSymbol))
      //          nSumOfCounts += m_fAlpha + nCounts(iParent * numValues + iSymbol)
      //        }
      //      }
      //
      //      if (nSumOfCounts != 0) {
      //        fLogScore -= logGamma(nSumOfCounts)
      //      }
      //
      //      if (m_fAlpha != 0) {
      //        fLogScore -= numValues * logGamma(m_fAlpha)
      //        fLogScore += logGamma(numValues * m_fAlpha)
      //      }

      //Bde
      //      var nSumOfCounts = 0.0
      //
      //      for (iSymbol <- 0 until numValues) {
      //        if (m_fAlpha + nCounts(iParent * numValues + iSymbol) != 0) {
      //          fLogScore += logGamma(1.0 / (numValues * nCardinality) + nCounts(iParent * numValues + iSymbol));
      //          nSumOfCounts += 1.0 / (numValues * nCardinality) + nCounts(iParent * numValues + iSymbol)
      //        }
      //      }
      //      fLogScore -= logGamma(nSumOfCounts);
      //
      //      fLogScore -= numValues * logGamma(1.0 / (numValues * nCardinality));
      //      fLogScore += logGamma(1.0 / nCardinality);

      //MDL,AIC,Entropy

      var nSumOfCounts = 0.0

      for (iSymbol <- 0 until numValues) {
        nSumOfCounts += nCounts(iParent * numValues + iSymbol)
      }

      for (iSymbol <- 0 until numValues) {
        if (nCounts(iParent * numValues + iSymbol) > 0) {
          fLogScore += nCounts(iParent * numValues +
            iSymbol) * Math.log(nCounts(iParent * numValues + iSymbol) / nSumOfCounts)
        }
      }

    }

    //MDL
    fLogScore -= 0.5 * nCardinality * (numValues - 1) * Math.log(data.size)
    //AIC
    //fLogScore -= nCardinality * (numValues - 1);

    fLogScore
  }

  def calcScoreWithExtraParent(nNode: Int, nCandidateParent: Int): Double = {
    val oParentSet = nodes(nNode).par;
    val contain = oParentSet.filter(node => node.i == nCandidateParent).size > 0
    if (contain) {
      -1e100;
    } else {
      // set up candidate parent
      oParentSet.+=(nodes(nCandidateParent))
      // calculate the score
      val logScore = calcNodeScore(nNode);
      // delete temporarily added parent
      nodes(nNode) = Node(nodes(nNode).i,
        oParentSet.filter(node => node.i != nCandidateParent),
        nodes(nNode).cpt, nodes(nNode).m_Distributions)
      logScore
    }
  }
  def indexInSet(str: String, sets: HashSet[String]): Int = {
    var i = 0
    var j = 0
    sets.map(it => { if (it.equalsIgnoreCase(str)) { i = j }; j += 1 })
    i
  }
  case class BNetModel() extends Model {

    def distributionForInstance(instance: LabeledFeature): String = {

      //Instances instances = bayesNet.m_Instances;
      var nNumClasses = idx(0).size
      var fProbs = Array.fill(nNumClasses)(1.0)

      for (label <- idx(0).toIterator) {
        val iClass = indexInSet(label, idx(0))
        var logfP = 0.0;

        for (iAttribute <- 0 until idx.size) {
          var iCPT = 0

          for (iParent <- 0 until nodes(iAttribute).par.size) {
            var nParent = nodes(iAttribute).par(iParent).i;

            if (nParent == 0) {
              iCPT = iCPT * nNumClasses + iClass;
            } else {
              iCPT = iCPT * idx(nParent).size +
                indexInSet(instance.features(nParent), idx(nParent));
            }
          }

          def getProbability(data: Int): Double = {
            val dist = nodes(iAttribute).m_Distributions(iCPT)
            if (dist.m_SumOfCounts == 0) {
              // this can only happen if numSymbols = 0 in constructor
              0.0
            } else {
              dist.m_Counts(data) / dist.m_SumOfCounts;
            }
          }

          if (iAttribute == 0) {

            logfP += Math.log(getProbability(iClass))
          } else {

            logfP += Math.log(getProbability(
              indexInSet(instance.features(iAttribute), idx(iAttribute))))
          }
        }

        //      fProbs[iClass] *= fP;
        fProbs(iClass) += logfP
      }

      // Find maximum
      var fMax = fProbs(0)
      var arr = idx(0).toArray
      var maxLabel = arr(0)
      for (iClass <- 0 until nNumClasses) {
        if (fProbs(iClass) > fMax) {
          fMax = fProbs(iClass)
          val arr = idx(0).toArray
          maxLabel = arr(iClass)
        }
      }
      // transform from log-space to normal-space
      for (iClass <- 0 until nNumClasses) {
        fProbs(iClass) = Math.exp(fProbs(iClass) - fMax)
      }

      // Display probabilities
      //Utils.normalize(fProbs);

      val all = fProbs.sum
      fProbs = fProbs.map(f => f / all)
      println(instance.label + "=>" + maxLabel + "=>" + fProbs.mkString(","))
      maxLabel
    }
    def predict(test: Instances): Double = {
      var r = 0.0

      val testdata = test.data.map(f => {
        val label = f.target
        var features = (ArrayBuffer(label) ++ f.features).toArray
        (LabeledFeature(label, features.map(f => f.trim())))
      })

      testdata.foreach(f => {
        val label = distributionForInstance(f)
        if (label.equalsIgnoreCase(f.label)) r += 1.0
      })
      r / test.data.size
    }
  }
}

object BayesNet {

  def main(args: Array[String]): Unit = {
    var numIdx = new HashSet[Int]
    numIdx.+=(0)
    numIdx.+=(1)
    numIdx.+=(2)
    numIdx.+=(3)
    //    numIdx.+=(5)
    //    numIdx.+=(7)
    //    numIdx.+=(8)
    //    numIdx.+=(10)
    val insts = new Instances(numIdx)
    insts.read("E:/books/spark/ml/decisionTree/iris.csv")
    insts.all2Normal
    val (trainset, testset) = insts.stratify()

    val t = new BayesNet(trainset)
    val model = t.train()

    val accu = model.predict(testset)
    println(accu);
  }

}