package algorithm.boost

import scala.collection.mutable.HashMap
import scala.collection.mutable.HashSet
import algorithm.Instances
import scala.Array.canBuildFrom

object Adaboost {

  def M1(insts: Instances, M: Int) {
    //insts.intWeight()
    val Gm: Array[(Int, Double, String, String, String, Double)] = Array.fill(M)(null)
    var isreturn = false
    var i = 0
    while (i < M && !isreturn) {
      insts.intWeight()
      val m = BaseClassfier.M1(insts)
      println(m.toString)
      val err = m._6
      if (err > 0.5) {
        println(i + ": err >" + err)
        isreturn = true
      } else {
        Gm(i) = m
        insts.reWeight(m)
      }
      i += 1
    }
    val Gmf = Gm.slice(0, i)

    instancesFor(insts, Gmf)

  }

  def instancesFor(insts: Instances, Gm: Array[(Int, Double, String, String, String, Double)]) {
    val data = insts.data
    data.map(f => {
      val l = f.label

      val dist = HashMap[String, Double]()
      Gm.map(gm => {

        val iAttr = gm._1
        val alpha = gm._2
        val splitFor = gm._3

        if (insts.numIdx.contains(iAttr)) {
          val split = splitFor.toDouble
          val col = f.features(iAttr)
          val ismis = col.equalsIgnoreCase("?")
          if (!ismis) {

            if (col.toDouble > split) {
              dist(gm._5) = dist.getOrElse(gm._5, 0.0) + gm._2
            } else {
              dist(gm._4) = dist.getOrElse(gm._4, 0.0) + gm._2
            }
          }
        } else {
          val split = splitFor
          val col = f.features(iAttr)
          val ismis = col.equalsIgnoreCase("?")
          if (!ismis) {

            if (col.equalsIgnoreCase(split)) {
              dist(gm._4) = dist.getOrElse(gm._4, 0.0) + gm._2
            } else {
              dist(gm._5) = dist.getOrElse(gm._5, 0.0) + gm._2
            }
          }
        }

      })
      val sum = dist.values.sum
      val nor = dist.map(f => {
        (f._1, f._2 / sum)

      })
      println(l + "=>" + nor)
    })

  }

  def M2(insts: Instances, M: Int) {
    insts.intWeight
    val data = insts.data
    val iClass = data.map(f => f.label).toSet

    var Wi_notyi = Array.fill(data.size)(new HashMap[String, Double]())
    //var i = 0
    data.map(f => {
      val label = f.label
      val w = f.weight
      val i = f.i
      iClass.map(cl => {
        if (!cl.equalsIgnoreCase(label)) { Wi_notyi(i)(cl) = w / (iClass.size - 1) }
      })
      // i += 1
    })
    val Gm: Array[(Int, Double, String, Map[String, Double], Map[String, Double], Double)] = Array.fill(M)(null)

    for (t <- 0 until M) {
      val Wti = Wi_notyi.map(l => l.values.sum).sum
      //
      //      var i =0 
      //      var Qtiy = Array.fill(data.size)(new HashMap[String, Double]())
      Wi_notyi = Wi_notyi.map(t => {
        t.map(f => {
          (f._1, f._2 / Wti)
        })
        //       i+=1
      })

      //      var sum = 0.0
      //      for (i <- 0 until data.size) {
      //        sum += data(i).weight
      //      }
      //Dti
      //      for (i <- 0 until data.size) {
      //        data(i).weight = data(i).weight/sum
      //      }

      val m = M2Classfier.M2(insts, /**Qtiy,*/ Wi_notyi)
      Gm(t) = m
    }
    instancesFor2(insts, Gm)
  }

  def instancesFor2(insts: Instances, Gm: Array[(Int, Double, String, Map[String, Double], Map[String, Double], Double)]) {
    val data = insts.data
    data.map(f => {
      val l = f.label

      val dist = HashMap[String, Double]()
      Gm.map(gm => {

        val iAttr = gm._1
        val alpha = gm._2
        val splitFor = gm._3

        if (insts.numIdx.contains(iAttr)) {
          val split = splitFor.toDouble
          val col = f.features(iAttr)
          val ismis = col.equalsIgnoreCase("?")
          if (!ismis) {

            val g = if (col.toDouble > split) { gm._5 } else { gm._4 }
            //dist(l) = dist.getOrElse(l, 0.0) + g(l) * alpha
            g.map(f => { dist(f._1) = dist.getOrElse(f._1, 0.0) + f._2 * math.log(1 / alpha) })
            //            if (col.toDouble > split) {
            //              gm._5.map(f => { dist(f._1) = dist.getOrElse(f._1, 0.0) + f._2 * math.log(1 / gm._2) })
            //              //dist(gm._5) = dist.getOrElse(gm._5, 0.0) + gm._2
            //            } else {
            //              gm._4.map(f => { dist(f._1) = dist.getOrElse(f._1, 0.0) + f._2 * math.log(1 / gm._2) })
            //              //dist(gm._4) = dist.getOrElse(gm._4, 0.0) + gm._2
            //            }
          } else {
            //gm._5.map(f => { dist(f._1) = dist.getOrElse(f._1, 0.0) + f._2 * math.log(1 / gm._2) })
            //gm._4.map(f => { dist(f._1) = dist.getOrElse(f._1, 0.0) + f._2 * math.log(1 / gm._2) })
          }
        } else {
          val split = splitFor
          val col = f.features(iAttr)
          val ismis = col.equalsIgnoreCase("?")
          if (!ismis) {
            val g = if (col.equalsIgnoreCase(split)) { gm._4 } else { gm._5 }
            //dist(l) = dist.getOrElse(l, 0.0) + g(l) * alpha
            g.map(f => { dist(f._1) = dist.getOrElse(f._1, 0.0) + f._2 * math.log(1 / alpha) })

            //            if (col.equalsIgnoreCase(split)) {
            //              //dist(gm._4) = dist.getOrElse(gm._4, 0.0) + gm._2
            //              
            //              //gm._4.map(f => { dist(f._1) = dist.getOrElse(f._1, 0.0) + f._2 * math.log(1 / gm._2) })
            //            } else {
            //              //dist(gm._5) = dist.getOrElse(gm._5, 0.0) + gm._2
            //              //gm._5.map(f => { dist(f._1) = dist.getOrElse(f._1, 0.0) + f._2 * math.log(1 / gm._2) })
            //            }
          } else {

            //gm._5.map(f => { dist(f._1) = dist.getOrElse(f._1, 0.0) + f._2 * math.log(1 / gm._2) })
            //gm._4.map(f => { dist(f._1) = dist.getOrElse(f._1, 0.0) + f._2 * math.log(1 / gm._2) })

          }
        }

      })
      val sum = dist.values.sum
      val nor = dist.map(f => {
        (f._1, f._2 / sum)

      })
      println(l + "=>" + nor)
    })

  }

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
    val insts = new Instances(numIdx)
    insts.read("E:/books/spark/ml/decisionTree/labor.csv")

    M1(insts, 21)
  }

}