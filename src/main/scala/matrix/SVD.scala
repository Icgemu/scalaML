package matrix

import Mat._

case class SVD(U: Matrix, S: Matrix, V: Matrix)

/**
 * tool for svd decomposition
 * http://www.ling.ohio-state.edu/~kbaker/pubs/Singular_Value_Decomposition_Tutorial.pdf
 * http://www.uwlax.edu/faculty/will/svd/index.html
 * http://www.cs.utexas.edu/users/inderjit/public_papers/HLA_SVD.pdf
 */
object SVD {
  
  private[this] def decomposition(B: Array[Array[Double]],
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
      B = T(A)
      decomposition(B, V, S, U)
    } else {
      decomposition(B, U, S, V)
    }
    SVD(U, r.S, V)
  }

}