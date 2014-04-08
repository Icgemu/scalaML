package algorithm.tree

import algorithm.Instances
import scala.collection.mutable.HashMap
import algorithm.tree.RFtree.Node
import scala.collection.mutable.HashSet

object RandomForest {
  
  def classifier(insts:Instances,T:Int,m:Int){
    
    val trees = Array.fill(T)(HashMap[Int,Node]())
    for(t<- 0 until T){
      val N = insts.data.size
      
      val X = Array.fill(N)(Array.fill(insts.attr)(""))
      val Y = Array.fill(N)("")
      for(n<- 0 until N){
        val i = (math.random * N).toInt
        X(n) = insts.data(i).features
        Y(n) = insts.data(i).label
      }
      
      val r = insts.attr - m
      val idx = insts.idx.map(f=>f)
      while(idx.size>m){
        val keys = idx.keys.toArray
        val t = (keys.size * math.random ).toInt
        idx.remove(keys(t)) 
      }
      
      val data = new Instances(insts.numIdx)
      data.read(X, Y)
      data.idx = idx
      
      val nodes=RFtree.classifier(data)
      RFtree.printTree(nodes, nodes(1), 0)
      trees(t) = nodes
    }
    test(insts,trees)
  }
  def test(insts:Instances,
      Fkm:Array[HashMap[Int,Node]]){
    val numIdx = insts.numIdx
    val Ks = insts.classof.toArray
    insts.data.map(lf => {
      var p = HashMap[String, Double]()
      //for(k<- Ks.toIterator){
      //p(k) += pk(lf,Ks,k,Fk0,Fkm,numIdx)
      Fkm.map(bt => {
        //val b = bt._1
        val f = bt
        val m = RFtree.instanceFor(f, 1, lf.features, numIdx)
        m.map(f=>{
          p(f._1) = p.getOrElse(f._1, 0.0) + f._2 
        })     
      })
      //}

      //       }
      val sum = p.values.sum
      val pro = p.map(f => { (f._1, f._2 * 1.0 / sum) })
      for (k <- Ks.toIterator) {
        print(k + "=" + pro.getOrElse(k, 0.0) + ",")
      }
      println(lf.label)
    })

  }

  def main(args: Array[String]): Unit = {
     var numIdx = new HashSet[Int]
    numIdx.+=(0)
    numIdx.+=(1)
    numIdx.+=(2)
    numIdx.+=(3)
    //    numIdx.+=(4)
    numIdx.+=(5)
    numIdx.+=(8)
    numIdx.+=(10)
    val insts = new Instances(numIdx)
    insts.read("E:/books/spark/ml/decisionTree/labor.csv")

    classifier(insts, 10,10)
  }

}