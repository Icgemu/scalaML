package matrix

import scala.collection.mutable.HashSet

object Mat {

  type Matrix = Array[Array[Double]]
  case class LU(L: Matrix, U: Matrix, P: Matrix)
  case class QR(Q: Matrix, R: Matrix)

  def T(mat: Matrix): Matrix = {
    val a = Array.ofDim[Double](mat(0).length, mat.length)
    for (i <- 0 until mat.length) {
      for (j <- 0 until mat(i).length) {
        a(j)(i) = mat(i)(j)
      }
    }
    return a
  }

  def I(m: Int): Matrix = {
    val mat = Array.ofDim[Double](m, m)
    for (i <- 0 until m) {
      mat(i)(i) = 1
    }
    mat
  }

  //行列式
  //|x|= det(x)
  def det(x: Matrix): Double = {
    val f = lu(x)
    val L = f.L
    val U = f.U
    var r = 1.0
    for (i <- 0 until L.length) { r += r * L(i)(i) }
    for (i <- 0 until U.length) { r += r * U(i)(i) }
    r
  }
  //scale
  def multi(x: Matrix, y: Double): Matrix = {

    val max = Array.fill(x.size)(Array.fill(x(0).size)(0.0))
    max.zip(x).map(row => {
      val r1 = row._1
      val r2 = row._2
      for (i <- 0 until r2.size) {
        r1(i) = r2(i) * y
      }
    })
    max
  }
  //矩阵加法
  def plus(x: Matrix,
    y: Matrix): Matrix = {
    val max = Array.fill(x.size)(Array.fill(x(0).size)(0.0))
    assert(x.length == y.length)
    assert(x(0).length == y(0).length)

    for (i <- 0 until x.length) {
      for (j <- 0 until x(i).length) {
        max(i)(j) = x(i)(j) + y(i)(j)
      }
    }
    max
  }
  //逆矩阵
  //高斯-约旦法
  def reverse(x: Matrix): Matrix = {
    val row = x.length
    val col = x(0).length

    val matrix = randomMatrix(row, col)

    var matr = x.map(row => {
      val t = row ++: (Array.fill(col)(0.0))
      t
    })
    assert(row == col) //只能解方阵

    for (i <- 0 until row) { //扩展一个单位矩阵I矩阵
      for (j <- row until 2 * col) {
        if (i + row == j) matr(i)(j) = 1 else matr(i)(j) = 0
      }
    }

    //开始矩阵变换
    for (fi <- 0 until row) {
      var max = matr(fi)(fi)
      var mrow = fi
      var mcol = fi

      //矩阵只进行初等行变换，每一列找出对角线下最大元
      //做这个的目的是保证akk != 0
      for (si <- fi until row; sj = fi) {
        if (max < matr(si)(sj)) {
          max = matr(si)(sj)
          mrow = si
          mcol = sj
        }
      }
      //主行与找到最大的主元的行进行交换  
      for (ti <- 0 until 2 * col) {
        if (mrow != fi) {
          //double *temp=new(nothrow) double[2*column];
          var temp = Array.fill(2 * col)(0.0)
          temp(ti) = matr(fi)(ti)
          matr(fi)(ti) = matr(mrow)(ti) / max
          matr(mrow)(ti) = temp(ti)
        } else {
          matr(fi)(ti) = matr(fi)(ti) / max
        }
      }

      for (li <- 0 until row) {
        val flag = matr(li)(fi) //设定标志，不至于它的值改变后影响
        if (li != fi) {
          for (lj <- 0 until 2 * col) {
            matr(li)(lj) = matr(li)(lj) - (matr(fi)(lj) * flag)
          }
        }
      }
    }
    //var 
    for (r <- 0 until matr.length) {
      for (c <- row until 2 * col) {
        matrix(r)(c - col) = matr(r)(c)
        //print(matr(r)(c) + "   ")
      }
      //println
    }
    matrix
  }

  def randomMatrix(m: Int, n: Int): Matrix = {

    val ma = Array.fill(m)(Array.fill(n)(0.0))
    ma.map(row => {

      for (i <- 0 until row.size) {
        row(i) = math.random
      }
    })
    ma
  }
  def randomorthoMatrix(m: Int): Matrix = {

    val ma = Array.fill(m)(Array.fill(m)(0.0))
    //ma.map(row=>{
    for (j <- 0 until m) {
      for (i <- 0 until m) {
        val r = math.random
        ma(i)(j) = r
        ma(j)(i) = r
      }
    }
    ma
  }

  //乘
  def dot(x: Matrix,
    y: Matrix): Matrix = {
    assert(x(0).length == y.length)
    val row = x.length
    val col = y(0).length
    var index = 0
    val r = new Array[Array[Double]](row).map(r => {

      val t = Array.fill(col)(0.0)
      val xrow = x(index)

      for (i <- 0 until t.length) {

        var sum = 0.0
        for (j <- 0 until y.length) {
          sum += xrow(j) * y(j)(i)
        }
        t(i) = sum
      }
      index += 1
      t
    })
    r
  }

  //Doolittle algorithm
  def lu(A: Matrix): LU = {
    val Row = A.length
    val Col = A(0).length

    assert(Row == Col)

    var L = new Array[Array[Double]](Row)
    //val U = new Array[Array[Double]](Row)

    var U = A.clone
    for (i <- 0 until Col) {

      val Li = new Array[Array[Double]](Row).map(r => {
        Array.fill(Col)(0.0)
      })
      for (k <- 0 until Row) {
        Li(k)(k) = 1.0
      }
      var r = 0
      val aii = U(i)(i)

      for (r <- i + 1 until Row) {
        Li(r)(i) = -1.0 * U(r)(i) / aii
      }
      U = dot(Li, U)
      L = if (L(0) == null) reverse(Li) else dot(L, reverse(Li))
    }
    LU(L, U, null)
  }

  def lup(A: Matrix): LU = {
    var P = Array.fill(A.length)(Array.fill(A(0).length)(0.0))
    var has = new HashSet[Int]()
    for (i <- 0 until A.length) {
      for (j <- 0 until A(0).length) {
        if (i == j) {
          P(i)(j) = 1
          if (A(i)(j) == 0) {
            var change = false
            for (t <- 0 until A.length) {
              if (t != i && !has.contains(t)) {
                if (A(t)(j) != 0) {
                  val temp = A(i).clone
                  A(i) = A(t)
                  A(t) = temp
                  has += i

                  val temp1 = P(i).clone
                  P(i) = P(t)
                  P(t) = temp1
                  change = true
                }
              }
            }
            assert(change == true)
          }

        } else {
          P(i)(j) = 0
        }
      }
    }

    val res = lu(A)

    LU(res.L, res.U, P)
  }

  /**
   * QR decomposition Using the Gram–Schmidt process
   * http://en.wikipedia.org/wiki/QR_decomposition
   */
  def qr(A: Matrix): QR = {

    val m = A.length
    val n = A(0).length
    val p = if (m > n) n else m

    var a_vector = T(A)

    val u = new Array[Array[Double]](a_vector.length)
    val e = new Array[Array[Double]](a_vector.length)

    def I(v: Array[Double]): Array[Double] = {
      val multi = v.map(ai => ai * ai)
      val sum = multi.reduce(_ + _)
      v.map(f => f / math.sqrt(sum))
    }

    u(0) = a_vector(0)
    e(0) = I(u(0))

    def proj(e: Array[Double], a: Array[Double]): Array[Double] = {
      val e1 = e.map(ai => ai * ai).reduce(_ + _)
      val a1 = e.zip(a).map(t => t._1 * t._2).reduce(_ + _)
      e.map(f => f * a1 / e1)
    }

    def derivevertor(a: Array[Double], b: Array[Double]): Array[Double] = {
      a.zip(b).map(t => t._1 - t._2)
    }

    for (i <- 1 until a_vector.length) {
      val ai = a_vector(i)
      var ui = a_vector(i)
      for (j <- 0 until i) {
        ui = derivevertor(ui, proj(e(j), ai))
      }
      u(i) = ui
      e(i) = I(ui)
    }

    val Q = T(e)
    val R = Array.fill(e.length)(Array.fill(a_vector.length)(0.0))

    def mutilvertor(a: Array[Double], b: Array[Double]): Double = {
      a.zip(b).map(t => t._1 * t._2).reduce(_ + _)
    }

    for (i <- 0 until R.length) {
      for (j <- i until R(i).length) {
        R(i)(j) = mutilvertor(e(i), a_vector(j))
      }
    }

    QR(Q, R)
  }

  /**
   * QR decomposition Using Householder reflections
   * http://en.wikipedia.org/wiki/QR_decomposition
   */
  def householder(A: Matrix): QR = {
    var cache = A
    val m = A.length
    val n = A(0).length
    //var a_vector =  transponse(A)

    //向量范数
    //Euclidean norm
    def norm(v: Array[Double]): Double = {
      val multi = v.map(ai => ai * ai)
      val sum = multi.reduce(_ + _)
      math.sqrt(sum)
    }

    def Mii(A: Matrix): Option[Matrix] = {

      if (A.length == 1 || A(0).length == 1) None else {

        var m = Array.fill(A.length - 1)(Array.fill(A(0).length - 1)(0.0))
        for (i <- 1 until A.length) {
          for (j <- 1 until A(i).length) {
            m(i - 1)(j - 1) = A(i)(j)
          }
        }
        Some(m)
      }
    }

    var p = math.min(m - 1, n)
    val IMatrix = I(m)
    var Qset = new Array[Matrix](p)

    var R = A
    var a_vector = T(R)
    for (i <- 0 until p) {
      //var a_vector =  R
      var ei = Array.fill(a_vector(0).length)(0.0)
      ei(0) = 1.0
      def sgn(x: Double): Double = if (x > 0) 1 else -1

      val alpha = -1 * sgn(R(i)(i)) * norm(a_vector(0))

      val ui = a_vector(0).zip(ei).map(t => t._1 + alpha * t._2)
      val norm_ui = norm(ui)
      val vi = ui.map(f => f / norm_ui)

      val v = Array[Array[Double]](vi)
      val vt = T(v)
      val aaa = multi(dot(vt, v), -2)
      val Qi = plus(I(m - i), aaa)

      var Q = I(Qi.length + i)
      for (j <- 0 until Qi.length) {
        for (k <- 0 until Qi(i).length) {
          Q(j + i)(k + i) = Qi(j)(k)
        }
      }

      Qset(i) = Q
      R = dot(Q, R)
      var temp = R
      var isover = false;
      for (k <- 0 to i) {
        val ge = Mii(temp)
        ge match {
          case None => isover = true
          case Some(t) => temp = t
        }
      }
      if (!isover) {
        a_vector = T(temp)
      }
      //a_vector = (Mii(R).get)
    }
    var Q: Array[Array[Double]] = T(Qset(0))
    for (i <- 1 until Qset.length) {
      Q = dot(Q, T(Qset(i)))
    }

    QR(Q, R)
  }

  /**
   * QR decomposition Using Givens Rotation
   * http://en.wikipedia.org/wiki/QR_decomposition
   */
  def givensRotation(A: Matrix): QR = {

    val m = A.length
    val n = A(0).length
    var R = A
    //var Gset = new ArrayBuffer[Array[Array[Double]]]()
    var Qt: Array[Array[Double]] = null
    for (j <- 0 until n) {
      for (k <- 1 until m - j; i = m - k) {
        if (A(i)(j) != 0) {
          val g = I(m)
          val a = R(i - 1)(j)
          val b = R(i)(j)
          val r = math.hypot(a, b)
          //NaN
          val c = a / r
          val s = -b / r

          g(i)(i) = c
          g(i - 1)(i - 1) = c
          g(i - 1)(i) = -s
          g(i)(i - 1) = s

          if (Qt == null) (Qt = g) else Qt = dot(g, Qt)
          R = dot(g, R)
        }
      }
    }
    val Q = T(Qt)
    QR(Q, R)
  }
}