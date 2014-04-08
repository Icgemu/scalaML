package algorithm.markev

import scala.Array.canBuildFrom

object HMM {

  val O = Map("A" -> 0, "B" -> 1, "C" -> 2, "D" -> 3, "E" -> 4, "F" -> 5, "G" -> 6, "H" -> 7, "I" -> 8, "J" -> 9,
    "K" -> 10, "L" -> 11, "M" -> 12, "N" -> 13, "O" -> 14, "P" -> 15, "Q" -> 16, "R" -> 17, "S" -> 18, "T" -> 19,
    "U" -> 20, "V" -> 21, "W" -> 22, "X" -> 23, "Y" -> 24, "Z" -> 25)
  //val O = Map("Dry"->0, "Dryish"->1, "Damp"->2, "Soggy"->3)

  case class HMM(Pi: Array[Double],
    A: Array[Array[Double]],
    B: Array[Array[Double]])
  def initAlpha(
    Alpha: Array[Array[Double]],
    Pi: Array[Double],
    A: Array[Array[Double]],
    B: Array[Array[Double]],
    Ot: Array[String]) {
    //Alpha init 0  
    for (i <- 0 until Alpha(0).length) {
      Alpha(0)(i) = 1e4 * Pi(i) * B(i)(O(Ot(0)))
    }

    for (t <- 1 until Ot.length) {
      for (j <- 0 until Alpha(t).length) {
        //for(j<- 0 until Alpha.length){
        var i = 0
        Alpha(t)(j) = Alpha(t - 1).map(f => { val r = f * A(i)(j) * B(j)(O(Ot(t))); i += 1; r }).sum
        //}
      }
    }
  }

  //计算观察序列的概率（Finding the probability of an observed sequence）
  def forwardAlg(
    Pi: Array[Double],
    A: Array[Array[Double]],
    B: Array[Array[Double]],
    Ot: Array[String]): Double = {
    var Alpha = Array.fill(Ot.length)(Array.fill(A.length)(0.0))
    initAlpha(Alpha, Pi, A, B, Ot)
    val PrYt = Alpha(Ot.length - 1).sum
    //PrYt
    //防止underflow
    math.log(PrYt)
  }

  def initGama(
    gama: Array[Array[Double]],
    Phi: Array[Int],
    Pi: Array[Double],
    A: Array[Array[Double]],
    B: Array[Array[Double]],
    Ot: Array[String]) {
    //Alpha init 0  
    for (i <- 0 until gama(0).length) {
      gama(0)(i) = 1e4 * Pi(i) * B(i)(O(Ot(0)))
    }

    for (t <- 1 until Ot.length) {
      for (j <- 0 until gama(t).length) {
        var i = 0
        val temp = gama(t - 1).map(f => { val r = f * A(i)(j) * B(j)(O(Ot(t))); i += 1; r })
        for (i <- 0 until temp.length) {
          val f = temp(i)
          if (f > gama(t)(j)) { gama(t)(j) = f; Phi(t - 1) = i }
        }

      }
    }
    val l = gama.last
    for (i <- 0 until l.length) {
      val f = l(i)
      var max = 0.0
      if (f > max) { max = f; Phi(Ot.length - 1) = i }
    }
  }
  //寻找最可能的隐藏状态序列(Finding most probable sequence of hidden states)
  def viterbi(
    Pi: Array[Double],
    A: Array[Array[Double]],
    B: Array[Array[Double]],
    Ot: Array[String]) {
    var gama = Array.fill(Ot.length)(Array.fill(A.length)(0.0))
    var Phi = Array.fill(Ot.length)(0)
    initGama(gama, Phi, Pi, A, B, Ot)

    for (i <- 0 until Phi.length - 1) {
      print("S" + Phi(i) + "=>")
    }
    println("S" + Phi.last)
  }
  def initBeta(
    Beta: Array[Array[Double]],
    Pi: Array[Double],
    A: Array[Array[Double]],
    B: Array[Array[Double]],
    Ot: Array[String]) {
    //Alpha init 0  
    for (i <- 0 until Beta(0).length) {
      Beta(Ot.length - 1)(i) = 1 * 1e4
    }

    for (k <- 0 until Ot.length - 1; t = Ot.length - 2 - k) {
      for (i <- 0 until Beta(t).length) {
        //for(j<- 0 until Alpha.length){
        var j = 0
        Beta(t)(i) = Beta(t + 1).map(bt => { val r = bt * A(i)(j) * B(j)(O(Ot(t + 1))); j += 1; r }).sum
        //}
      }
    }
  }
  def initY(
    Y: Array[Array[Double]],
    Alpha: Array[Array[Double]],
    Beta: Array[Array[Double]]) {
    val temp = Alpha.zip(Beta).map(t => t._1.zip(t._2).map(k => k._1 * k._2))
    val sum = temp.map(t => t.sum)
    for (i <- 0 until temp.length) {
      Y(i) = temp(i).map(t => t / sum(i))
    }
  }

  def baumWelch(
    Pi: Array[Double],
    A: Array[Array[Double]],
    B: Array[Array[Double]],
    Ot: Array[String]): HMM = {
    var a1 = forwardAlg(Pi, A, B, Ot)
    var iscov = false
    while (!iscov) {
      var Beta = Array.fill(Ot.length)(Array.fill(A.length)(0.0))
      var Alpha = Array.fill(Ot.length)(Array.fill(A.length)(0.0))
      var Y = Array.fill(Ot.length)(Array.fill(A.length)(0.0))
      initBeta(Beta, Pi, A, B, Ot)
      initAlpha(Alpha, Pi, A, B, Ot)
      initY(Y, Alpha, Beta)
      var eta = Array.fill(Ot.length)(Array.fill(A.length)(Array.fill(A.length)(0.0)))
      for (t <- 0 until eta.length - 1) {
        var sum = 0.0
        for (i <- 0 until A.length) {
          for (j <- 0 until A.length) {
            eta(t)(i)(j) = Alpha(t)(i) * A(i)(j) * B(j)(O(Ot(t + 1))) * Beta(t + 1)(j)
            sum += eta(t)(i)(j)
          }
        }
        for (i <- 0 until A.length) {
          for (j <- 0 until A.length) {
            eta(t)(i)(j) = eta(t)(i)(j) / sum
          }
        }
      }

      for (i <- 0 until Pi.length) { Pi(i) = Y(0)(i) }
      for (i <- 0 until A.length) {
        for (j <- 0 until A.length) {
          var sum1 = 0.0
          var sum2 = 0.0
          for (t <- 0 until Ot.length - 1) {
            sum1 += eta(t)(i)(j)
            sum2 += Y(i)(j)
          }
          A(i)(j) = sum1 / sum2
        }
        for (k <- 0 until B(i).length) {
          var sum1 = 0.0
          var sum2 = 0.0
          for (t <- 0 until Ot.length) {
            if (O(Ot(t)) == k) { sum1 += Y(t)(i) }
            sum2 += Y(t)(i)
          }
          B(i)(k) = sum1 / sum2
        }
      }
      var a2 = forwardAlg(Pi, A, B, Ot)
      if (a2 - a1 < 0.01) {
        iscov = true
      } else {
        a1 = a2
      }
    }
    HMM(Pi, A, B)
  }

  def main(args: Array[String]): Unit = {
    val M = 26
    val N = 10
    var Pi = Array.fill(N)(1.0 / N)
    // Pi = Array(0.63, 0.17, 0.20)
    var A = Array.fill(N)(Array.fill(N)(1.0 / N))
    //    A(0)=Array(0.500, 0.375, 0.125)
    //    A(1)=Array(0.250 ,0.125, 0.625)
    //    A(2)=Array(0.250, 0.375, 0.375)
    var B = Array.fill(N)(Array.fill(M)(1.0 / M))
    //    B(0)=Array(0.60, 0.20, 0.15, 0.05)
    //    B(1)=Array(0.25, 0.25, 0.25, 0.25)
    //    B(2)=Array(0.05, 0.10, 0.35, 0.50)
    //      B(0)=Array(0.50, 0.5)
    //      B(1)=Array(0.75, 0.25)
    //      B(2)=Array(0.25, 0.75)

    val Ot = Array("A", "C", "F", "G", "R", "E", "W", "H", "L", "B", "T")
    //val Ot = Array("Dry","Dry","Dry","Dry","Dryish","Dry","Dryish","Dryish","Dryish","Dryish")
    //val data = loadCsv()
    //predict(data,cluster(data,3))
    //forwardAlg(Pi,A,B,Ot)
    //viterbi(Pi,A,B,Ot)
    baumWelch(Pi, A, B, Ot)
  }

}