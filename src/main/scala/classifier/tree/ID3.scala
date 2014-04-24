package classifier.tree

import classifier.Classifier
import classifier.Model
import core.Instances
import scala.collection.mutable.ArrayBuffer
import core.Feature
import scala.collection.mutable.HashMap
import scala.collection.mutable.HashSet

class ID3(
  insts: Instances) extends TreeClassifierBase {

  case class Node(i: Int, sets: HashMap[String, Node], label: String)

  def train(): Model = {

    var idx = HashMap[Int, HashSet[String]]()

    insts.data.map(f => {
      val features = f.features
      for (i <- 0 until features.length) {
        val v = features(i)
        if (!idx.contains(i)) { idx.put(i, new HashSet[String]()) }
        if (!idx(i).contains(v)) { idx(i) += v.trim() }
      }
    })

    ID3Model(train0(insts.data, idx))
  }

  def train0(data: ArrayBuffer[Feature], idx: HashMap[Int, HashSet[String]]): Node = {
    val filters = ratio(data).filter(t => t._2 > 0.95).toArray
    if (filters.size > 0) {
      Node(-1, null, filters(0)._1)
    } else {

      //entropy for this data sets
      val e_entropy = exp_entropy(data)
      //condition entropy for every feature
      val c_entropy = new HashMap[Int, Double]

      for (i <- idx.keys) {
        
        c_entropy(i) = cond_entropy(data.groupBy(f=>f.features(i)))
      }

      //find max gain 
      var gains = c_entropy.map(f => { (f._1, e_entropy - f._2) })
      var maxGain = (-1.0)
      var maxGainIndex = (-1)
      for (i <- gains.keys) {
        if (gains(i) > maxGain) { maxGain = gains(i); maxGainIndex = i }
      }

      idx.remove(maxGainIndex)
      if(idx.size>0){
	      var sets = new HashMap[String, Node]()
	
	      //split the data sets
	      var splitData = new HashMap[String, ArrayBuffer[Feature]]
	      data.map(f => {
	        var ty = f.features(maxGainIndex)
	        if (!splitData.contains(ty)) { splitData.put(ty, new ArrayBuffer[Feature]) }
	        splitData(ty) += f
	      })
	
	      splitData.map(f => {
	        sets(f._1) = train0(f._2, idx)
	      })
	      Node(maxGainIndex, sets, "")
      }else{
           val r = ratio(data).toArray
           val max = r.sortBy(f=>f._2).reverse
           val label = max(0)._1
         
    	  Node(-1, null, label)
      }   
    }
  }

  case class ID3Model(nodes: Node) extends Model {
    def predict(test: Instances): Double = {
      var r = 0.0
      test.data.map(lf => {
        var node = nodes
        var label = ""
        while (node.i > 0) {
          label = node.label
          node = node.sets(lf.features(node.i))
        }
        label = node.label
        if (label.equalsIgnoreCase(lf.target)) r += 1.0

      })
      r / test.data.size
    }
  }
}
object ID3 {

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
    insts.all2Normal
    val (trainset,testset) = insts.stratify()
    
    val t = new ID3(trainset)    
    val model = t.train()
    
    val accu = model.predict(testset)
    println(accu);
  }
}