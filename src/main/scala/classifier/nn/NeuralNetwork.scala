package classifier.nn

import classifier.Classifier
import classifier.Model
import core.Instances
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashSet

//http://ufldl.stanford.edu/wiki/index.php/Backpropagation_Algorithm
class NeuralNetwork(
  insts: Instances,
  learningRate: Double,
  iteration: Int,
  hiddenLayerNodesNum: Array[Int]) extends Classifier {

  val inputSize = insts.attr + 1 //add 1 as attribute 0
  val outputSize = insts.classof.size
  // unit count for every layers
  var unitNum_forLayer = ArrayBuffer[Int]()
  unitNum_forLayer += inputSize
  unitNum_forLayer ++= (hiddenLayerNodesNum.map(f => f + 1))
  unitNum_forLayer += outputSize

  //theta=> layer l+1 to layer l weight
  var theta = unitNum_forLayer.map(f => {
    val v = new Array[Array[Double]](f)
    v
  })

  //gradient=> gradient sum for each theta in one iteration.
  var gradient = unitNum_forLayer.map(f => {
    val v = new Array[Array[Double]](f)
    v
  })

  //var i = 0;
  //theta(l)(i)(j)=>layer l's the ith unit has a link with layer (l-1)'s the jth unit
  //gradient(l)(i)(j)=>layer l's the ith unit has a link with layer (l-1)'s the jth unit
  for (layer <- 0 until (theta.length - 1)) {

    val a = theta(layer)
    val b = theta(layer + 1)
    //if (layer < theta.length - 2) {
    for (j <- 0 until b.length) {
      theta(layer + 1)(j) = new Array[Double](a.length)
      gradient(layer + 1)(j) = new Array[Double](a.length)
      for (i <- 0 until a.length) {
        theta(layer + 1)(j)(i) = math.random - 0.5
        gradient(layer + 1)(j)(i) = 0.0
      }
    }
   

  }

  //output of layers
  var a_forLayer = unitNum_forLayer.map(f => {
    val v = Array.fill(f)(1.0)
    v
  })
 
  //error of units
  //gama(l)(j) => layer l's the jth unit
  var gama = unitNum_forLayer.map(f => {
    val v = Array.fill(f)(1.0)
    val t = v.map(f => math.random)
    t
  })

  def train(): Model = {

    for (ite <- 0 until iteration) {
      
      insts.data.map(lf => {
        val x = Array(1.0, lf.features.map(f => f.toDouble): _*)
        a_forLayer(0) = x
        val y = insts.classToInt(lf.target)
        // forward algorithm
        for (layer <- 1 until a_forLayer.length) {
          val l = a_forLayer(layer)
          var j = 0
          val notlast = layer < (a_forLayer.length - 1)
          a_forLayer(layer) = l.map(f => {
            val ai = if (j == 0 && notlast) 1.0 else {

              var v = theta(layer)(j).zip(a_forLayer(layer - 1))
              val r = v.map(p => {
                p._1 * p._2
              }).reduce(_ + _)

              //sigmoid activation function
              //1 / (1 + math.exp(-r - bias_forLayer(i)(j)))
              1 / (1 + math.exp(-r))
            }
            j = j + 1
            ai
          })
        }
        
        //BP algorithm
        val last = a_forLayer.last
        var no = 0
        //last layer's gama = (ai - yi)
        gama(gama.length - 1) = last.map(f => {
          var ret = 0.0
//                  if (y == no) {
//                    ret = (1 - f) * (f * (1 - f))
//                  } else {
//                    ret = (0 - f) * (f * (1 - f))
//                  }
          if (y == no) {
            ret = (1 - f)
          } else {
            ret = (0 - f)
          }        
          no = no + 1
          -1.0 * ret
        })
        //gama backward
        for (i <- 1 until gama.length - 1) {
          val layer = gama.length - 1 - i
          val units = gama(layer).length;
          val l1 = gama(layer + 1).length;

          for (k <- 1 until units) {
            gama(layer)(k) = {
              var sum = 0.0
              val out = if (i == 1) 0 else 1
              for (t <- out until l1) {
                sum += theta(layer + 1)(t)(k) * gama(layer + 1)(t)
                //sum+=theta(j+1)(t)(k)*gama(j+1)(t)*a_forLayer(j+1)(t)*(1-a_forLayer(j+1)(t))
              }
              //val err = if(i==1)sum else sum*a_forLayer(j)(k)*(1-a_forLayer(j)(k))
              val err = sum * a_forLayer(layer)(k) * (1 - a_forLayer(layer)(k))
              //bias_forLayer(j)(k) = bias_forLayer(j)(k) + 0.01 * err            
              err
            }
          }
        }
        
        //renew gradient
        for (i <- 1 until gama.length) {
          val layer = gama.length - i
          val units = gama(layer).length;
          val l1 = gama(layer - 1).length;

          for (k <- 1 until units) {
            //
            for (j <- 0 until l1) {
              gradient(layer)(k)(j) = gradient(layer)(k)(j) + gama(layer)(k) * a_forLayer(layer - 1)(j)
            }
          }
        }
      })

      
      //batch renew theta
      for (layer <- 1 until theta.length) {
        //theta(i)={
        var l = theta(layer)
        val out = if (layer == theta.length - 1) 0 else 1
        for (i <- out until l.length) {
          val t = theta(layer)(i)
          //val out = if(i== theta.length-1) 0 else 1
          for (j <- 0 until t.length) {
            if (j == 0) {
              theta(layer)(i)(j) = theta(layer)(i)(j) - learningRate * (gradient(layer)(i)(j) / insts.data.size)
            } else {
              theta(layer)(i)(j) = theta(layer)(i)(j) - learningRate * (gradient(layer)(i)(j) / insts.data.size + 0.01 * theta(layer)(i)(j))
            }
            gradient(layer)(i)(j) = 0.0
          }
        }
      }
    }
    NNModel()
  }

  case class NNModel() extends Model {

    override def predict(test: Instances): Double = {

      var r = 0.0
      test.data.map(lf => {
        a_forLayer(0) = Array(1.0, lf.features.map(f => f.toDouble): _*)
        for (i <- 1 until a_forLayer.length) {
          val l = a_forLayer(i)
          var j = 0
          val notlast = i < (a_forLayer.length - 1)
          a_forLayer(i) = l.map(f => {
            val result = if (j == 0 && notlast) 1.0 else {
              var v = theta(i)(j).zip(a_forLayer(i - 1))
              val r = v.map(p => {
                p._1 * p._2
              }).reduce(_ + _)
              1 / (1 + math.exp(-r))          
            }
            j = j + 1
            result
          })
        }
        var last = a_forLayer.last
        val sum = last.sum
        last = last.map(_ / sum)
        val max = last.max
        val t = insts.classToInt(lf.target)
        if (last(t) == max) { r += 1.0 }
        System.out.print(lf.target+"=>(")
        for (i <- insts.classof) {
          System.out.print(i+"=>"+last(insts.classToInt(i))+ ",")
        }
        System.out.println(")")
      })
      r / test.data.size
    }
  }
}
object NeuralNetwork {

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
    val t = new NeuralNetwork(insts, 0.15, 1000, Array(8))
    val model = t.train()
    val accu = model.predict(insts)
    println(accu);
  }
}