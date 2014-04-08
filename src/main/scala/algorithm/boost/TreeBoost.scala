package algorithm.boost

import algorithm.Instances
import algorithm.LabeledFeature
import scala.collection.mutable.HashMap
import algorithm.RegInstances
import algorithm.tree.CARTReg
import scala.collection.mutable.ArrayBuffer
import algorithm.tree.CARTReg.Node
import scala.collection.mutable.HashSet
import algorithm.RegFeature

//Gradient Boosting Tree
object TreeBoost {
  
  def classifier(insts:RegInstances,M:Int,J:Int) = {
    //val K = insts.classof.size
    //val Ks = insts.classof.toArray
    
    insts.intWeight
    var Fkm = ArrayBuffer[HashMap[Int,Node]]()
    var Fk0 = 0.0
    //Ks.map(f=>Fk0(f) = 0.0)
    var is = true
    var y = Double.MaxValue
    while(is){
      //sum = y
      println("MM----------------MM")
      //val tmp = HashMap[String,HashMap[Int,Node]]()
      //for(iclass<- Ks.iterator){
        //val k = insts.classToInt(iclass)
        val X = Array.fill(insts.data.size)(Array.fill(insts.attr)(""))
        val Y = Array.fill(insts.data.size)(0.0)
        var sum = 0.0
        for(i<- 0 until insts.data.size){
          val inst= insts.data(i)
          
          X(i) = inst.features                  
          val c = pk(inst,i/**Ks,iclass*/,Fk0,Fkm,insts.numIdx)
          //println(p)
          //if(i==0){println(inst.features.mkString(",")+"=>"+c)}
          val yik = inst.weight - c
          sum += yik * yik
          //inst.weight = yik
          //注意是梯度的负方向
          //Y(i) = -1*yik
         // println(yik)
          Y(i) = yik
        }
      println("Log="+sum)
      if(y - sum < 100){
        is = false
      }else{
        y = sum
      }
        val reg = new RegInstances(insts.numIdx)
        reg.read(X, Y)
        //reg.data.map(f=>println(f.features.mkString(",")+","+f.value))
        val nodes = CARTReg.classifier(reg, J)
        CARTReg.printTree(nodes,nodes(1), 0)
        //tmp(iclass) = nodes
      //}
      Fkm += nodes
    }
    
    test(insts,Fk0,Fkm)
  }

  def pk(x:RegFeature,/**Ks:Array[String],iclass:String,*/
      i:Int,
      Fk0:Double,
      Fkm:ArrayBuffer[HashMap[Int,Node]],numIdx:HashSet[Int]): Double = {
    var fork = 0.0
    //var sum = 0.0
    //for(i<- Ks.toIterator){
      //println("---")
      val t = (Fkm.map(f=>{
        val nodes = f
        //CARTReg.printTree(nodes(1), 0)
        val c = CARTReg.instanceFor(nodes, 1, x.features, numIdx)
        ///println(x.features.mkString(",")+"==>"+c)
        //val r =((Ks.size-1)*1.0/Ks.size)*c
        //if(i==0)println(c)
        val r = c
        r
      }).sum + Fk0)
      //println(t)
      //sum += math.exp(t)
      //println(t +"=>" +sum)
//      if(i.equalsIgnoreCase(iclass)){
//        fork = t//math.exp(t)
//      }
    //}
    //println(fork+"/"+sum)
//    if(sum<1e-10){
//      0.0
//    }else{
//      fork/sum
//    }
    //fork
      
      t
  }
  
  def test(insts:RegInstances,
      Fk0:Double,
      Fkm:ArrayBuffer[HashMap[Int,Node]]){
    val numIdx = insts.numIdx
    //val Ks = insts.classof.toArray
    insts.data.map(lf=>{
//      val p = HashMap[String,Double]()
//       for(k<- Ks.toIterator){
         //p(k) = pk(lf,Ks,k,Fk0,Fkm,numIdx)
      val c = pk(lf,-1,Fk0,Fkm,numIdx)
//       }
//      for(k<- Ks.toIterator){
//         print(k+"="+p(k)+",")
//       }
      println(lf.value + "=>"+ c)
    })
   
  }
  
  def main(args: Array[String]): Unit = {
    var numIdx = new HashSet[Int]
    numIdx.+=(0)
    numIdx.+=(1)
    numIdx.+=(2)
    numIdx.+=(3)
    numIdx.+=(4)
    numIdx.+=(5)
//    numIdx.+=(8)
//    numIdx.+=(10)
    val insts = new RegInstances(numIdx)
    insts.read("E:/books/spark/ml/decisionTree/cpu.csv")

    classifier(insts, 30,6)
  }

}