package cluster

import scala.collection.mutable.ArrayBuffer
import core.Feature

object SimpleKMeans {

  case class KMean(K: Int,
    centers: Array[Array[Double]],
    par: Map[Int, ArrayBuffer[Feature]])

  def kmean(data: ArrayBuffer[Feature], K: Int): KMean = {
    var k = K
    var partition = data.groupBy(f => math.round((math.random * (k - 1))).toInt)
    var centers = for (i <- 0 until k) yield data((data.size * math.random).toInt).features.map(f => f.toDouble)

    var isCov = false
    var lastDist = Double.MaxValue
    var ite = 0
    while (!isCov) {
      print("itarate " + ite)
      ite += 1
      var totalDist = 0.0
      val temp = data.groupBy(d => {
        var index = 0
        var dist = Double.MaxValue
        val fe = d.features.map(f => f.toDouble)
        for (i <- 0 until centers.length) {
          val f = centers(i).zip(fe).map(t => {
            (t._1 - t._2) * (t._1 - t._2)
          }).reduce(_ + _)
          totalDist += f
          if (f < dist) { dist = f; index = i }

        }
        index
      })
      println("=>" + math.abs(totalDist - lastDist))
      if (math.abs(totalDist - lastDist) < 0.1) {
        isCov = true
      } else {
        var newC = temp.map(f => {
          val size = f._2.size

          var sum = f._2.map(s => {
            val t1s = s.features.map(f => f.toDouble)
            //val t2s = s._2.features.map(f=>f.toDouble)
            //t1s.zip(t2s).map(t=>{t._1 + t._2})
            t1s
          }).reduce((t1, t2) => {
            t1.zip(t2).map(t => { t._1 + t._2 })
          })
          sum.map(f => f / size)
        })
        partition = temp
        centers = newC.toIndexedSeq
        lastDist = totalDist
      }
    }
    KMean(k, centers.toArray, partition)
  }
  def main(args: Array[String]): Unit = {}

}