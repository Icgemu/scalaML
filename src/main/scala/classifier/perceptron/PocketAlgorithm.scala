package classifier.perceptron

import core.Instances
import scala.collection.mutable.ArrayBuffer
import core.Feature
import scala.collection.mutable.HashSet


//http://ftp.cs.nyu.edu/~roweis/csc2515-2006/readings/gallant.pdf
//perceptron-based learning algorithms
class PocketAlgorithm(
  insts: Instances,
  learningRate: Double,
  iteration: Int,
  cvFold: Int) extends StdPerceptron(insts, learningRate, iteration, cvFold) {

  override def train0(
    data: ArrayBuffer[Feature],
    w: Array[Double],
    bias: Double,
    pair: (String, String)): InterModel = {
    
    var pi = Array(bias, w: _*)
    var run_pi, run_W, num_ok_pi, num_ok_W = 0
    var W = pi
    var isbreak = false
    var j = iteration
    while (!isbreak && j > 0) {
      val ramSel = data((data.size * math.random).toInt)
      val yi = if (ramSel.target.equalsIgnoreCase(pair._1)) 1 else -1
      val fs = ramSel.features.map(f => f.toDouble)
      var f = Array(1.0, fs: _*)

      val rst = f.zip(pi).map(t => { t._1 * t._2 }).sum

      if ((rst > 0 && yi == 1) || (rst <= 0 && yi == (-1))) {
        run_pi += 1
        if (run_pi > run_W) {

          num_ok_pi = 0
          data.map(lf => {
            val yi = if (lf.target.equalsIgnoreCase(pair._1)) 1 else -1
            val fs = lf.features.map(f => f.toDouble)
            var f = Array(1.0, fs: _*)

            val rst = f.zip(pi).map(t => { t._1.toDouble * t._2 }).sum
            if ((rst > 0 && yi == 1) || (rst <= 0 && yi == (-1))) {
              num_ok_pi += 1
            }
          })
          if (num_ok_pi > num_ok_W) {
            W = pi
            run_W = run_pi
            num_ok_W = num_ok_pi
          }
          if (num_ok_W == data.size) {
            isbreak = true
          }

        }
      } else {

        pi = pi.zip(f).map(t => { t._1 + yi * t._2 })
        run_W = 0
      }
      j -= 1
    }

    InterModel(W, pair)
  }
}

object PocketAlgorithm {

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
    val t = new PocketAlgorithm(insts, 0.05, 100, 10)
    val model = t.train()
    val accu = model.predict(insts)
    println(accu);
  }
}