package algorithm

object Base64 {

  val digits = Array[Char](
    '0', '1', '2', '3', '4', '5',
    '6', '7', '8', '9', 'a', 'b',
    'c', 'd', 'e', 'f', 'g', 'h',
    'i', 'j', 'k', 'l', 'm', 'n',
    'o', 'p', 'q', 'r', 's', 't',
    'u', 'v', 'w', 'x', 'y', 'z',
    'A', 'B', 'C', 'D', 'E', 'F',
    'G', 'H', 'I', 'J', 'K', 'L',
    'M', 'N', 'O', 'P', 'Q', 'R',
    'S', 'T', 'U', 'V', 'W', 'X',
    'Y', 'Z', '+', '/')

  /**
   * 把10进制的数字转换成64进制
   * @param number
   * @param shift 2的shift次方进制
   * @return
   */
  def CompressNumber(number: Long, shift: Int): String = {

    assert(shift < 7)

    var num = number
    val buf = Array.fill(64)('0')
    var charPos = 64
    val radix = 1 << shift
    val mask = radix - 1
    do {
      charPos -= 1
      buf(charPos) = digits((num & mask).toInt)
      num >>>= shift
    } while (num != 0)
    new String(buf, charPos, (64 - charPos))
  }

  /**
   * 把64进制的字符串转换成10进制
   * @param decompStr
   * @return
   */
  def UnCompressNumber(decompStr: String): Long = {
    var result = 0l

    result += getCharIndexNum(decompStr.last)
    val l = decompStr.length
    for (k <- 0 until l - 1; i = l - 2 - k) {

      for (j <- 0 until digits.length) {
        if (decompStr(i) == digits(j)) {
          result += (j.toLong) << 6 * (l - 1 - i)
        }
      }
    }
    result
  }

  /**
   *
   * @param ch
   * @return
   */
  def getCharIndexNum(ch: Char): Long =
    {
      val num = (ch).toInt

      val r = if (num >= 48 && num <= 57) {
        num - 48
      } else if (num >= 97 && num <= 122) {
        num - 87
      } else if (num >= 65 && num <= 90) {
        num - 29
      } else if (num == 43) {
        62
      } else if (num == 47) {
        63
      } else {
        0
      }
      r
    }
  def main(args: Array[String]): Unit = {
    //    val t = 1278
    //    val r = CompressNumber(t,6)
    //    println(r)
    //    val t1 = UnCompressNumber(r)
    //    println(t1)
    println(CompressNumber(605752100005L, 6))
    println(UnCompressNumber(CompressNumber(605752100005L, 6)))
    println(CompressNumber(636317L, 6))
    println(UnCompressNumber(CompressNumber(636317L, 6)))

  }
}