package algorithm.recommander

import scala.io.Source
import java.io.File
import scala.collection.mutable.HashSet
import scala.Array.canBuildFrom

object RecommenderSystem {

  case class Model(av:Double,
      bu:Array[Double],
      bi:Array[Double],
      pu:Array[Array[Double]],
      qi:Array[Array[Double]])
      
  def average(file: String): (Double, Int, Int) = {

    val buff = Source.fromFile(new File(file))
    var cnt = 0;
    var sum = 0.0
    var items = new HashSet[Int]()
    var users = new HashSet[Int]()
    buff.getLines.toArray.map(line => {
      cnt += 1
      val arr = line.split("	")
      sum += arr(2).toDouble
      items.+=(arr(1).toInt)
      users.+=(arr(0).toInt)
    })
    buff.close
    (sum / cnt, items.max, users.max)
  }

  def innerProduct(v1: Array[Double], v2: Array[Double]): Double = {
    v1.zip(v2).map(t => t._1 * t._2).reduce(_ + _)
  }

  def score(av: Double,
    bu: Double,
    bi: Double,
    pu: Array[Double],
    qi: Array[Double]): Double = {
    val sc = av + bu + bi + innerProduct(pu, qi)

    if (sc < 1) 1 else if (sc > 5) 5 else sc
  }

  def svd(trainFile: String, testFile: String):Model = {
    val av = average(trainFile)
    val averageScore = av._1
    val userNum = av._3
    val itemNum = av._2

    val factorNum = 10
    val learnRate = 0.01
    val regularization = 0.05

    var bi = Array.fill(itemNum)(0.0)
    var bu = Array.fill(userNum)(0.0)

    val temp = math.sqrt(factorNum)

    var qi = Array.fill(itemNum)(Array.fill(factorNum)((0.1 * math.random / temp)))
    var pu = Array.fill(userNum)(Array.fill(factorNum)((0.1 * math.random / temp)))

    //train model
    var preRmse = 1000000.0
    var step = 0
    //var isbreak = true
    while (step <52) {
      println("Step " + step)
      val buff = Source.fromFile(new File(trainFile))
      buff.getLines.toArray.map(line => {
        val arr = line.split("	")
        val iid = arr(1).toInt -1
        val uid = arr(0).toInt -1
        val sc = arr(2).toDouble

        val r = score(averageScore, bu(uid), bi(iid), pu(uid), qi(iid))

        val eui = sc - r

        //update parameters
        bu(uid) += learnRate * (eui - regularization * bu(uid))
        bi(iid) += learnRate * (eui - regularization * bi(iid))

        for (k <- 0 until factorNum) {
          val temp = pu(uid)(k) //attention here, must save the value of pu before updating
          pu(uid)(k) += learnRate * (eui * qi(iid)(k) - regularization * pu(uid)(k))
          qi(iid)(k) += learnRate * (eui * temp - regularization * qi(iid)(k))
        }
      })
      buff.close
//      val curRmse = validate(testFile, averageScore, bu, bi, pu, qi)
//      if(curRmse > preRmse){
//        isbreak = true
//      }else{
//        preRmse = curRmse
        step +=1
//      }
//      println(curRmse)
    }

    Model(averageScore,bu,bi,pu,qi)
  }

  def validate(testFile:String,
      av:Double,
      bu:Array[Double],
      bi:Array[Double],
      pu:Array[Array[Double]],
      qi:Array[Array[Double]]):Double = {
    
    var cnt =0
    var rmse = 0.0
    val buff = Source.fromFile(new File(testFile))
    buff.getLines.toArray.map(line=>{
      val arr = line.split("	")
      val iid = arr(1).toInt -1
      val uid = arr(0).toInt -1 
      val sc = arr(2).toDouble

      val r = score(av, bu(uid), bi(iid), pu(uid), qi(iid))

      rmse += (sc - r) * (sc - r)
    })
    buff.close
    rmse
  }
  
  def predict(model:Model,testFile:String){
    val buff = Source.fromFile(new File(testFile))
    buff.getLines.toArray.map(line=>{
      val arr = line.split("	")
      val iid = arr(1).toInt -1
      val uid = arr(0).toInt -1 
      val sc = arr(2).toDouble

      val r = score(model.av, model.bu(uid), model.bi(iid), model.pu(uid),model.qi(iid))

      print(line+"==>" + r)
      println
    })
    buff.close
  }
  def main(args: Array[String]): Unit = {
    val model = svd("E://books/spark/ml/SVD/ml_data/training.txt",
        "E://books/spark/ml/SVD/ml_data/test.txt")
    predict(model,"E:/books/spark/ml/SVD/ml_data/test.txt")
  }

}