package algorithm.regression

import algorithm.Instances
import algorithm.matrix.MatrixDecomposition._
import scala.collection.mutable.HashSet
import scala.collection.mutable.ArrayBuffer

object LDA {

  def Lda(insts: Instances) {

    val data = insts.data
    val attrs = insts.attr
    val avgAttr = Array.fill(attrs)(0.0)
    data.map(f => {
      val lf = f.features.map(t => t.toDouble)
      for (i <- 0 until lf.size) {
        avgAttr(i) = avgAttr(i) + lf(i)
      }
    })
    val avg = avgAttr.map(f => { f / data.size })

    val tran = data.map(lf => {
      val l = lf.features.map(t => t.toDouble)
      val tmp = Array.fill(attrs)(0.0)
      for (i <- 0 until l.size) {
        tmp(i) = l(i) - avg(i)
      }
      tmp
    }).toArray

    var cov = Array.fill(attrs)(Array.fill(attrs)(0.0))

    for (i <- 0 until attrs) {
      val a1 = transpose(Array.fill(1)(tran(i)))
      for (j <- 0 until attrs) {
        val a2 = (Array.fill(1)(tran(j)))
        val t = dot(a1, a2)
        cov = plus(cov, t)
      }
    }
    cov = multi(cov, 1.0 / data.size)

    val de = svd(cov)

    val u = de.U //特征向量
    val s = de.S // 特征值
    //val v = de.V

    val vset = HashSet[Int]()
    for (i <- 0 until attrs) {
      if (s(i)(i) > 1e-10) {
        vset += (i)
      }
    }
    val uT = transpose(u)
    val UT = new ArrayBuffer[Array[Double]]()
    vset.map(f => UT += (uT(f)))
    val U = (UT.toArray)

    //数据降维
    val X = tran.map(lf => {

      val f = lf //lf.features.map(f=>f.toDouble)
      val xi = transpose(Array.fill(1)(f))
      val zi = transpose(dot(U, xi))(0)
      zi
    }).toArray

    val Y = data.map(lf => {

      lf.label
    }).toArray

    val ins = new Instances(insts.numIdx)
    ins.read(X, Y)
    //降维后分类
    LogisticRegression.classifier(insts, 80, 0.1, 10)

    //近似还原
    //xi = dot(U,zi)
  }

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

    Lda(insts)
  }

}