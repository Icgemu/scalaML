package classifier.tree

import classifier.Classifier
import scala.collection.mutable.ArrayBuffer
import core.Feature
import utils.Discretization

trait TreeClassifierBase extends Classifier {

  def log(f: Double): Double = {
    math.log(f) / math.log(2)
  }
  /**
   * data sets D's entropy H(D)
   */
  def exp_entropy(data: ArrayBuffer[Feature]): Double = {
    val total = data.map(f => f.weight).sum
    val map = data.groupBy(f => f.target)
    val scale = map.map(lb => (lb._1, lb._2.map(f => f.weight).sum / total))
    -1.0 * scale.values.map(f => f * log(f)).sum
  }

  /**
   * data sets D's feature i's condition entropy H(D|A)
   */
  def cond_entropy(data: Map[String,ArrayBuffer[Feature]]): Double = {
    val total = data.values.flatMap(f => f.map(lf=>lf.weight)).sum
//    val map = if (split.size > 0) {
//      data.groupBy(lf => {
//        val f = lf.features(i).toDouble
//        if (f > split.head) { "2" } else { "1" }
//      })
//    } else {
//      data.groupBy(f => f.features(i))
//    }

    val scale = data.map(lb => (lb._1, lb._2.map(f => f.weight).sum / total))
    scale.map(f => f._2 * exp_entropy(data(f._1))).sum
  }

  /**
   * data sets D's feature i's information gain G(D,Ai)
   */
  def gain_entropy(data: Map[String,ArrayBuffer[Feature]]): Double = {
    val all = ArrayBuffer(data.values.flatMap(f=>f).toArray:_*)
    exp_entropy(all) - cond_entropy(data)
  }

  /**
   * data sets D's feature i's information gain ration Gi(D,Ai)/Hai(D)
   */
  def ratio_entropy(data: Map[String,ArrayBuffer[Feature]]): Double = {
    val total = data.values.flatMap(f => f.map(lf=>lf.weight)).sum

//    val map = if (split.size > 0) {
//      data.groupBy(lf => {
//        val f = lf.features(i).toDouble
//        if (f > split.head) { "2" } else { "1" }
//      })
//    } else {
//      data.groupBy(f => f.features(i))
//    }

    val scale = data.map(lb => (lb._1, lb._2.map(f => f.weight).sum / total))
    val HaD = -1.0 * scale.values.map(f => f * log(f)).sum
    gain_entropy(data) / HaD
  }
  /**
   * data sets D's feature i's information gain Hai(D)
   */
  def ha_entropy(data: Map[String,ArrayBuffer[Feature]]): Double = {
    val total = data.values.flatMap(f => f.map(lf=>lf.weight)).sum

//    val map = if (split.size > 0) {
//      data.groupBy(lf => {
//        val f = lf.features(i).toDouble
//        if (f > split.head) { "2" } else { "1" }
//      })
//    } else {
//      data.groupBy(f => f.features(i))
//    }

    val scale = data.map(lb => (lb._1, lb._2.map(f => f.weight).sum / total))
    //print(scale.mkString(","))
    //println(":>" +(-1.0 * scale.values.map(f => f * log(f)).sum))
    -1.0 * scale.values.map(f => f * log(f)).sum
  }

  def ratio(data: ArrayBuffer[Feature]): Map[String, Double] = {
    val total = data.map(f => f.weight).sum
    val map = data.groupBy(f => f.target)
    map.map(lb => (lb._1, lb._2.map(f => f.weight).sum / total))
  }

  def hitAndMiss(data: ArrayBuffer[Feature], label: String): (Double, Double) = {
    val hit = data.filter(p => p.target.equalsIgnoreCase(label)).map(f => f.weight).sum
    val tol = data.map(f => f.weight).sum
    (hit, tol - hit)
  }
}