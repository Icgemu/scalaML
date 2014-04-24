package classifier

import scala.collection.mutable.HashSet
import scala.collection.mutable.ArrayBuffer
import core.Feature

trait ClassifierBase extends Classifier{
  
  /**
   * class label pair for multiple classification
   * using one vs one
   *
   *  @param cls all class labels set
   *  @return tuple(labelA,labelB) set
   */
  def makePair(cls: HashSet[String]): HashSet[(String, String)] = {
    val pair = new HashSet[(String, String)]
    cls.map(f => {
      cls.map(t => {
        if (t < f) pair.+=((t, f)) else pair.+=((f, t))
      })
    })
    val f = pair.filterNot(t => { t._1.equalsIgnoreCase(t._2) })
    f
  }

  /**
   * separate training set into cvFold
   *
   *  @param data training set
   *  @param cvFold number of fold;
   *  @return Array(ArrayBuffer[LabeledFeature])
   */
  def makeFold(data: ArrayBuffer[Feature],
    cvFold: Int): Array[ArrayBuffer[Feature]] = {

    val f = Array.fill(cvFold)(new ArrayBuffer[Feature])
    val r = if (data.size % cvFold == 0) data.size / cvFold else data.size / cvFold + 1
    for (i <- 0 until r) {
      for (j <- 0 until cvFold) {
        if ((i + j) < data.size)
          f(j).+=(data(i + j))
      }
    }
    f
  }
  /**
   * merge fold data except the i fold
   *
   *  @param f all fold set
   *  @param i this fold is not merged;
   *  @return ArrayBuffer[LabeledFeature]
   */
  def mergeFold(f: Array[ArrayBuffer[Feature]],
    i: Int): ArrayBuffer[Feature] = {
    var merge = new ArrayBuffer[Feature]()
    for (j <- 0 until f.size) {
      if (i != j) { merge ++= (f(j)) }
    }
    merge
  }
  /**
   * randomize the training data
   *
   *  @param data  training data
   *  @return ArrayBuffer[LabeledFeature]
   */
  def shuffle(data: ArrayBuffer[Feature]) {
    val size = data.size
    for (i <- 0 until size) {
      val r1 = (size * math.random).toInt
      val r2 = (size * math.random).toInt
      val r1f = data(r1)
      val r2f = data(r2)
      data(r1) = r2f
      data(r2) = r1f
    }
  }
  
  

}