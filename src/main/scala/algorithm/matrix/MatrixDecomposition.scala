package algorithm.matrix

import scala.collection.mutable.HashSet
import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuffer

object MatrixDecomposition {
  case class LU(L: Array[Array[Double]], U: Array[Array[Double]], P: Array[Array[Double]])
  case class QR(Q: Array[Array[Double]], R: Array[Array[Double]])

  case class SVD(U: Array[Array[Double]], S: Array[Array[Double]], V: Array[Array[Double]])

  //Doolittle algorithm
  def lu(A: Array[Array[Double]]): LU = {
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

  def lup(A: Array[Array[Double]]): LU = {
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

  def transpose(A: Array[Array[Double]]): Array[Array[Double]] = {
    val a = Array.fill(A(0).length)(Array.fill(A.length)(0.0))

    for (i <- 0 until A.length) {
      for (j <- 0 until A(i).length) {
        a(j)(i) = A(i)(j)
      }
    }

    a
  }

  //Using the Gram–Schmidt process
  def qr(A: Array[Array[Double]]): QR = {

    val m = A.length
    val n = A(0).length
    val p = if (m > n) n else m

    var a_vector = transpose(A)

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

    val Q = transpose(e)
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
  //单位矩阵
  def I(m: Int): Array[Array[Double]] = {
    var matrix = Array.fill(m)(Array.fill(m)(0.0))
    for (i <- 0 until m) {
      matrix(i)(i) = 1
    }
    matrix
  }
  //QR decomposition
  //Using Householder reflections
  def householder(A: Array[Array[Double]]): QR = {
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

    def Mii(A: Array[Array[Double]]): Option[Array[Array[Double]]] = {

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
    var Qset = new Array[Array[Array[Double]]](p)

    var R = A
    var a_vector = transpose(R)
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
      val vt = transpose(v)
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
        a_vector = transpose(temp)
      }
      //a_vector = (Mii(R).get)
    }
    var Q: Array[Array[Double]] = transpose(Qset(0))
    for (i <- 1 until Qset.length) {
      Q = dot(Q, transpose(Qset(i)))
    }

    QR(Q, R)
  }

  def hypot(x: Double, y: Double): Double = {
    var t = 0.0
    var x1 = math.abs(x)
    var y1 = math.abs(y)
    t = math.min(x1, y1)
    x1 = math.max(x1, y1)
    t = t / x1

    x1 * math.sqrt(1 + t * t);
  }
  //Givens Rotation
  //QR
  def givensRotation(A: Array[Array[Double]]): QR = {

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
    val Q = transpose(Qt)
    QR(Q, R)
  }

  def decomposition(B: Array[Array[Double]],
    U: Array[Array[Double]],
    S: Array[Double],
    V: Array[Array[Double]]): SVD = {
    var m = B.length
    var n = B(0).length

    //var p = math.min(m, n)

    var e = Array.fill(n)(0.0)
    var work = Array.fill(m)(0.0)

    val wantu = true
    val wantv = true

    var nct = math.min(m - 1, n)
    var nrt = math.max(0, n - 2)

    //val i, j, k = 0

    for (k <- 0 until math.max(nct, nrt)) {
      if (k < nct) {
        // Compute the transformation for the k-th column and
        // place the k-th diagonal in s[k].
        // Compute 2-norm of k-th column without under/overflow.

        for (i <- k until m) { S(k) = math.hypot(S(k), B(i)(k)) }
        if (S(k) != 0.0) {
          if (B(k)(k) < 0) { S(k) = -S(k) }
          for (i <- k until m) { B(i)(k) /= S(k) }
          B(k)(k) += 1
        }
        S(k) = -S(k)
      }
      for (j <- k + 1 until n) {
        if ((k < nct) && (S(k) != 0)) {
          // apply the transformation
          var t = 0.0
          for (i <- k until m)
            t += B(i)(k) * B(i)(j)

          t = -t / B(k)(k)
          for (i <- k until m)
            B(i)(j) += t * B(i)(k)
        }
        e(j) = B(k)(j)
      }

      // Place the transformation in U for subsequent back
      // multiplication.
      if (wantu && (k < nct)) {
        for (i <- k until m)
          U(i)(k) = B(i)(k)
      }

      if (k < nrt) {
        // Compute the k-th row transformation and place the
        // k-th super-diagonal in e(k].
        // Compute 2-norm without under/overflow.
        e(k) = 0;
        for (i <- k + 1 until n)
          e(k) = math.hypot(e(k), e(i))

        if (e(k) != 0) {
          if (e(k + 1) < 0)
            e(k) = -e(k);

          for (i <- k + 1 until n)
            e(i) /= e(k)
          e(k + 1) += 1
        }
        e(k) = -e(k)

        if ((k + 1 < m) && (e(k) != 0)) {
          // apply the transformation
          for (i <- k + 1 until m)
            work(i) = 0

          for (j <- k + 1 until n)
            for (i <- k + 1 until m)
              work(i) += e(j) * B(i)(j);

          for (j <- k + 1 until n) {
            var t = -e(j) / e(k + 1);
            for (i <- k + 1 until m)
              B(i)(j) += t * work(i);
          }
        }

        // Place the transformation in V for subsequent
        // back multiplication.
        if (wantv)
          for (i <- k + 1 until n)
            V(i)(k) = e(i);
      }
    }

    // Set up the final bidiagonal matrix or order p.
    var p = n

    if (nct < n)
      S(nct) = B(nct)(nct)
    if (m < p)
      S(p - 1) = 0

    if (nrt + 1 < p)
      e(nrt) = B(nrt)(p - 1)
    e(p - 1) = 0

    // if required, generate U
    if (wantu) {
      for (j <- nct until n) {
        for (i <- 0 until m)
          U(i)(j) = 0;
        U(j)(j) = 1;
      }

      for (temp <- 0 to nct - 1; k = nct - 1 - temp) {
        if (S(k) != 0) {
          for (j <- k + 1 until n) {
            var t = 0.0;
            for (i <- k until m)
              t += U(i)(k) * U(i)(j);
            t = -t / U(k)(k);

            for (i <- k until m)
              U(i)(j) += t * U(i)(k);
          }

          for (i <- k until m)
            U(i)(k) = -U(i)(k);
          U(k)(k) = 1 + U(k)(k);

          for (i <- 0 until k - 1)
            U(i)(k) = 0
        } else {
          for (i <- 0 until m)
            U(i)(k) = 0
          U(k)(k) = 1
        }
      }
    }

    // if required, generate V
    if (wantv) {
      for (temp <- 0 to n - 1; k = n - 1 - temp) {
        if ((k < nrt) && (e(k) != 0))
          for (j <- k + 1 until n) {
            var t = 0.0;
            for (i <- k + 1 until n)
              t += V(i)(k) * V(i)(j);
            t = -t / V(k + 1)(k);

            for (i <- k + 1 until n)
              V(i)(j) += t * V(i)(k);
          }

        for (i <- 0 until n)
          V(i)(k) = 0;
        V(k)(k) = 1;
      }
    }

    // main iteration loop for the singular values
    val pp = p - 1
    var iter = 0
    val eps = math.pow(2.0, -52.0)

    while (p > 0) {
      var k = 0
      var kase = 0

      // Here is where a test for too many iterations would go.
      // This section of the program inspects for negligible
      // elements in the s and e arrays. On completion the
      // variables kase and k are set as follows.
      // kase = 1     if s(p) and e[k-1] are negligible and k<p
      // kase = 2     if s(k) is negligible and k<p
      // kase = 3     if e[k-1] is negligible, k<p, and
      //				s(k), ..., s(p) are not negligible
      // kase = 4     if e(p-1) is negligible (convergence).
      var temp1 = p - 2
      var isbreak1 = false
      while (temp1 >= -1 && !isbreak1) //for( temp<-(-1) to p-2; k = p-2 -temp )
      {

        k = temp1
        if (k != -1) {
          if (math.abs(e(k)) <= eps * (math.abs(S(k)) + math.abs(S(k + 1)))) {
            e(k) = 0
            isbreak1 = true
          } else {
            temp1 -= 1
          }
        } else {
          isbreak1 = true
        }

      }

      if (k == p - 2) { kase = 4 }
      else {
        var ks = p - 1
        var isbreak = false
        while (ks >= k && !isbreak) {
          if (ks != k) {

            var t = (if (ks != p) math.abs(e(ks)) else 0) +
              (if (ks != k + 1) math.abs(e(ks - 1)) else 0)

            if (math.abs(S(ks)) <= eps * t) {
              S(ks) = 0
              isbreak = true
            } else {
              ks -= 1
            }
          } else {
            isbreak = true
          }

        }

        if (ks == k)
          kase = 3
        else if (ks == p - 1)
          kase = 1
        else {
          kase = 2
          k = ks
        }
      }
      k += 1

      // Perform the task indicated by kase.
      kase match {
        // deflate negligible s(p)
        case 1 =>
          {
            var f = e(p - 2)
            e(p - 2) = 0

            for (temp <- k to p - 2; j = p - 2 - temp + k) {
              var t = math.hypot(S(j), f)
              var cs = S(j) / t
              var sn = f / t
              S(j) = t

              if (j != k) {
                f = -sn * e(j - 1)
                e(j - 1) = cs * e(j - 1)
              }

              if (wantv)
                for (i <- 0 until n) {
                  t = cs * V(i)(j) + sn * V(i)(p - 1)
                  V(i)(p - 1) = -sn * V(i)(j) + cs * V(i)(p - 1)
                  V(i)(j) = t
                }
            }
          }

        // split at negligible s(k)
        case 2 =>
          {
            var f = e(k - 1)
            e(k - 1) = 0

            for (j <- k until p) {
              var t = math.hypot(S(j), f)
              var cs = S(j) / t
              var sn = f / t
              S(j) = t
              f = -sn * e(j)
              e(j) = cs * e(j)

              if (wantu)
                for (i <- 0 until m) {
                  t = cs * U(i)(j) + sn * U(i)(k - 1)
                  U(i)(k - 1) = -sn * U(i)(j) + cs * U(i)(k - 1)
                  U(i)(j) = t
                }
            }
          }

        // perform one qr step
        case 3 =>
          {
            // calculate the shift
            var scale = math.max(math.max(math.max(math.max(
              math.abs(S(p - 1)), math.abs(S(p - 2))), math.abs(e(p - 2))),
              math.abs(S(k))), math.abs(e(k)))
            var sp = S(p - 1) / scale
            var spm1 = S(p - 2) / scale
            var epm1 = e(p - 2) / scale
            var sk = S(k) / scale
            var ek = e(k) / scale
            var b = ((spm1 + sp) * (spm1 - sp) + epm1 * epm1) / 2.0
            var c = (sp * epm1) * (sp * epm1)
            var shift = 0.0

            if ((b != 0) || (c != 0)) {
              shift = math.sqrt(b * b + c)
              if (b < 0)
                shift = -shift
              shift = c / (b + shift)
            }
            var f = (sk + sp) * (sk - sp) + shift
            var g = sk * ek

            // chase zeros
            for (j <- k until p - 1) {
              var t = math.hypot(f, g)
              var cs = f / t
              var sn = g / t
              if (j != k)
                e(j - 1) = t

              f = cs * S(j) + sn * e(j)
              e(j) = cs * e(j) - sn * S(j)
              g = sn * S(j + 1)
              S(j + 1) = cs * S(j + 1)

              if (wantv) {
                for (i <- 0 until n) {
                  t = cs * V(i)(j) + sn * V(i)(j + 1)
                  V(i)(j + 1) = -sn * V(i)(j) + cs * V(i)(j + 1)
                  V(i)(j) = t
                }
              }

              t = math.hypot(f, g)
              cs = f / t
              sn = g / t
              S(j) = t
              f = cs * e(j) + sn * S(j + 1)
              S(j + 1) = -sn * e(j) + cs * S(j + 1)
              g = sn * e(j + 1)
              e(j + 1) = cs * e(j + 1)

              if (wantu && (j < m - 1)) {
                for (i <- 0 until m) {
                  t = cs * U(i)(j) + sn * U(i)(j + 1)
                  U(i)(j + 1) = -sn * U(i)(j) + cs * U(i)(j + 1)
                  U(i)(j) = t
                }
              }
            }
            e(p - 2) = f
            iter = iter + 1
          }

        // convergence
        case 4 =>
          {
            // Make the singular values positive.
            if (S(k) <= 0) {
              S(k) = if (S(k) < 0) -S(k) else 0
              if (wantv)
                for (i <- 0 to pp)
                  V(i)(k) = -V(i)(k)
            }

            // Order the singular values.
            var isbreak = false
            while (k < pp && !isbreak) {
              if (S(k) < S(k + 1)) {

                var t = S(k)
                S(k) = S(k + 1)
                S(k + 1) = t

                if (wantv && (k < n - 1)) {
                  for (i <- 0 until n) {
                    val t1 = V(i)(k)
                    V(i)(k) = V(i)(k + 1)
                    V(i)(k + 1) = t1
                  }
                }

                if (wantu && (k < m - 1)) {
                  for (i <- 0 until m) {
                    var t1 = U(i)(k)
                    U(i)(k) = U(i)(k + 1)
                    U(i)(k + 1) = t1
                  }
                }
                k += 1
              } else {
                isbreak = true
              }
            }
            iter = 0
            p -= 1
          }
      }
    }

    val N = S.length
    var T = Array.fill(N)(Array.fill(N)(0.0))
    for (i <- 0 until N) {
      T(i)(i) = S(i)
    }
    SVD(U, T, V)
  }
  def svd(A: Array[Array[Double]]): SVD = {

    var m = A.length
    var n = A(0).length
    var p = math.min(m, n)
    var U = Array.fill(m)(Array.fill(p)(0.0))
    var V = Array.fill(n)(Array.fill(p)(0.0))
    var S = Array.fill(p)(0.0)

    var B = A
    val r = if (m < n) {
      B = transpose(A)
      decomposition(B, V, S, U)
    } else {
      decomposition(B, U, S, V)
    }
    SVD(U, r.S, V)
  }
  //行列式
  //|x|= det(x)
  def det(x: Array[Array[Double]]): Double = {
    val f = lu(x)
    val L = f.L
    val U = f.U
    var r = 1.0
    for (i <- 0 until L.length) { r += r * L(i)(i) }
    for (i <- 0 until U.length) { r += r * U(i)(i) }
    r
  }
  //scale
  def multi(x: Array[Array[Double]], y: Double): Array[Array[Double]] = {

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
  def plus(x: Array[Array[Double]],
    y: Array[Array[Double]]): Array[Array[Double]] = {
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
  def reverse(x: Array[Array[Double]]): Array[Array[Double]] = {
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

  def randomMatrix(m: Int, n: Int): Array[Array[Double]] = {

    val ma = Array.fill(m)(Array.fill(n)(0.0))
    ma.map(row => {

      for (i <- 0 until row.size) {
        row(i) = math.random
      }
    })
    ma
  }
  def randomorthoMatrix(m: Int): Array[Array[Double]] = {

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
  def dot(x: Array[Array[Double]],
    y: Array[Array[Double]]): Array[Array[Double]] = {
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

  def main(args: Array[String]): Unit = {

    //    val x = new Array[Array[Double]](4)
    //    x(0) = Array(1.0, 0.0, 0.0,0,2)
    //    x(1) = Array(0.0, 0.0, 3.0,0,0)
    //    x(2) = Array(0.0, 0.0, 0.0,0,0)
    //    x(3) = Array(0.0, 4.0, 0.0,0,0)
    val x = new Array[Array[Double]](2)
    x(0) = Array(1.0, 3.0, 5.0, 7)
    x(1) = Array(2.0, 4.0, 6.0, 8)

    val t = svd(x)
    //val t1 = dot(transpose(t.Q), t.Q)
    val r = plus(x, multi(dot(dot(t.U, t.S), transpose(t.V)), -1))
    r
  }

}