package cluster
import matrix.Mat._
import scala.io.Source
import java.io.File
import scala.Array.canBuildFrom

object GMM {
  //K class  D Dimension N instances
  case class KMean(K: Int,
    centers: Array[Array[Double]],
    par: Map[Int, Array[Array[Double]]])
  case class GMM(
    pSigma: Array[Array[Array[Double]]],
    pMiu: Array[Array[Double]],
    pPi: Array[Double],
    K: Int)
  def kmean(data: Array[Array[Double]], K: Int): KMean = {
    var k = K
    var partition = data.groupBy(f => math.round((math.random * (k - 1))).toInt)
    var centers = for (i <- 0 until k) yield data((data.size * math.random).toInt)

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
        for (i <- 0 until centers.length) {
          val f = centers(i).zip(d).map(t => {
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

          var sum = f._2.reduce((t1, t2) => {
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

  def cluster(data: Array[Array[Double]],
    K: Int): GMM = {

    var Lpre = Double.MaxValue; // 上一次聚类的误差
    var threshold = 0.001

    val km = kmean(data, K)
    var centers = km.centers
    val par = km.par

    val D = data(0).length
    //各K类的之间的分配比例
    var pPi = par.map(f => f._2.size * 1.0 / data.size).toArray

    //K类里面每个特征的参数，即X前面的参数
    var pMiu = centers
    //协方差
    var pSigma = par.map(t => {
      val k = t._1
      val datak = t._2
      var sum = datak.reduce((t1, t2) => t1.zip(t2).map(k => k._1 + k._2)).map(d => d / datak.size)
      var sigma = Array.fill(D)(Array.fill(D)(0.0))
      for (i <- 0 until D) {
        var sigmai = sigma(i)
        for (j <- 0 until D) {
          var cov = 0.0
          datak.map(d => cov += (d(i) - sum(i)) * (d(j) * sum(j)))
          sigmai(j) = cov
        }
      }
      sigma
    }).toArray

    var isbreak = false
    while (!isbreak) {
      var px = prob(data, pSigma, pMiu, pPi)
      val sumpx = px.map(f => f.sum)
      px = px.zip(sumpx).map(t => t._1.map(f => f / t._2))

      val N = data.length
      val pxt = T(px).map(f => f.sum)

      for (k <- 0 until pxt.length) {
        val miuk = pMiu(k)
        val Nk = pxt(k)
        var sum = Array.fill(miuk.length)(0.0)
        var sum1 = Array.fill(miuk.length)(Array.fill(miuk.length)(0.0))
        for (i <- 0 until N) {
          val xi = data(i)
          sum = xi.map(f => f * px(i)(k)).zip(sum).map(t => t._1 + t._2)
          val xi_u = xi.zip(miuk).map(t => t._1 - t._2)
          sum1 = plus(multi(dot(T(Array(xi_u)), Array(xi_u)), px(i)(k)), sum1)
        }

        pMiu(k) = sum.map(f => f / Nk)
        pPi(k) = Nk / N
        pSigma(k) = multi(sum1, (1 / Nk))
      }

      var total = 0.0
      for (i <- 0 until px.length) {
        var sum = 0.0
        for (k <- 0 until pPi.length) {
          sum += px(i)(k) * pPi(k)
        }
        total += math.log(sum)
      }
      total = math.abs(total)
      if (math.abs(total - Lpre) < threshold) {
        isbreak = true
      } else {
        //if(total<Lpre){
        Lpre = total
        //}
      }
    }

    GMM(pSigma, pMiu, pPi, K)
  }

  def prob(data: Array[Array[Double]],
    pSigma: Array[Array[Array[Double]]],
    pMiu: Array[Array[Double]],
    pPi: Array[Double]): Array[Array[Double]] = {
    var px = Array.fill(data.size)(Array.fill(pPi.length)(0.0))
    for (i <- 0 until data.length) {
      for (k <- 0 until pPi.length) {
        val x_u = data(i).zip(pMiu(k)).map(t => t._1 - t._2)
        val inv = reverse(pSigma(k))
        val temp = math.exp(dot(dot((Array(x_u)), inv), T(Array(x_u)))(0)(0) * (-0.5))
        px(i)(k) = pPi(k) * math.pow((1 / math.sqrt(2 * math.Pi)), pPi.length) * math.sqrt(det(inv)) * temp
      }
    }
    px
  }

  def predict(data: Array[Array[Double]], model: GMM) {
    var px = prob(data, model.pSigma, model.pMiu, model.pPi)
    val sumpx = px.map(f => f.sum)
    px = px.zip(sumpx).map(t => t._1.map(f => f / t._2))
    for (i <- 0 until data.length) {
      var label = 1
      var max = -1.0
      for (j <- 0 until px(i).length) {
        if (px(i)(j) > max) {
          label = j + 1
          max = px(i)(j)
        }
      }
      println(data(i).toString() + "=>" + label)
    }
  }
  def loadCsv(): Array[Array[Double]] = {
    val f = Source.fromFile(new File("D://tmp/iris.csv"))
    val x = f.getLines.toArray.map(line => {
      val l = line.split(":")
      val data = l(0).split(",").map(v => v.toDouble)
      data
    })
    x
  }
  def main(args: Array[String]): Unit = {

    val data = loadCsv()
    predict(data, cluster(data, 3))
  }

}