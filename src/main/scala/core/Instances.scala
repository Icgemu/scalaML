package core

import scala.collection.mutable.HashSet
import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import java.io.File
import utils.Discretization

/**
 * simple Class for manipulating data info.
 * 
 * @author Icgemu
 * @version 0.0.1
 */
class Instances (attrIdxForNumeric:HashSet[Int],var isRegression:Boolean=false){
  
  //Nominal set for every attribute 
  //HashMap key :attribute index from 0 ~ ,value:each different Nominal
  //for Numeric attribute ,value is empty HashSet
  var idxForNominal = HashMap[Int, HashSet[String]]()
  //the real data
  var data = ArrayBuffer[Feature]()
  
  val numIdx = attrIdxForNumeric;
  
  //for classification only/all possible label
  var labels = new HashSet[String]
  
  //attribute size
  def attr(): Int = idxForNominal.size
  //retrieve the specify attribute.
  def attr(i: Int): HashSet[String] = idxForNominal(i)
  //the size of different class 
  def numClass(): Int = labels.size
  //retrieve all class labels
  def classof(): HashSet[String] = labels
  
  //retrieve class index for Feature
  
  def classToInt(lf: Feature): Int = {
    if(isRegression) return -1
    val label = lf.target
    var i = 0
    var j = 0
    labels.map(f => { if (f.equalsIgnoreCase(label)) { j = i }; i += 1 })
    j
  }
  
  //retrieve class index for the specify target
  def classToInt(lf: String): Int = {
    if(isRegression) return -1
    val target = lf
    var i = 0
    var j = 0
    labels.map(f => { if (f.equalsIgnoreCase(target)) { j = i }; i += 1 })
    j
  }
  
  //read file,the target must the last
  def read(file: String) {
    var buff = Source.fromFile(new File(file))
    var j = 0
    for(line <- buff.getLines) {
      val arr = line.split(",")
      val target = arr.last.trim()
      if(!isRegression)labels += target
      val features = arr.slice(0, arr.length - 1)
      for (i <- 0 until features.length) {
        val v = features(i)
        if (!idxForNominal.contains(i)) { idxForNominal.put(i, new HashSet[String]()) }
        //if attribute is Nominal then add this value to HashSet,else skip.
        if (!idxForNominal(i).contains(v) && !numIdx.contains(i)) { idxForNominal(i) += v.trim() }
      }
      //the default weight is 1.0
      data.+=(new Feature(j, target, features.map(f => f.trim()), 1.0))
      j += 1
    }
    buff.close
  }
  
  /**
   * read data for memory
   * 
   * @param x attribute
   * @param y target
   */ 
  def read(x: Array[Array[Double]], y: Array[Int]) {
    var j = 0
    x.map(line => {
      val arr = line
      val target = y(j)
      if(!isRegression)labels += (target + "")
      val features = arr.slice(0, arr.length)
      for (i <- 0 until features.length) {
        var v = features(i)
        if (!idxForNominal.contains(i)) { idxForNominal.put(i, new HashSet[String]()) }
        if (!idxForNominal(i).contains(v + "") && !idxForNominal.contains(i)) { idxForNominal(i) += (v + "") }
      }
      data.+=(new Feature(j, target + "", features.map(f => f + ""), 1.0))
      j += 1
    })
  }
  
  def read(x: Array[Array[Double]], y: Array[String]) {
    var j = 0
    x.map(line => {
      val arr = line
      val target = y(j)
      if(!isRegression)labels += (target + "")
      val features = arr.slice(0, arr.length)
      for (i <- 0 until features.length) {
        var v = features(i)
        if (!idxForNominal.contains(i)) { idxForNominal.put(i, new HashSet[String]()) }
        if (!idxForNominal(i).contains(v + "") && !numIdx.contains(i)) { idxForNominal(i) += (v + "") }
      }
      data.+=(new Feature(j, target, features.map(f => f + ""), 1.0))
      j += 1
    })
  }
  
  def read(x: Array[Array[String]], y: Array[String]) {
    var j = 0
    x.map(line => {
      val arr = line
      val target = y(j)
      if(!isRegression)labels += (target + "")
      val features = arr.slice(0, arr.length)
      for (i <- 0 until features.length) {
        var v = features(i)
        if (!idxForNominal.contains(i)) { idxForNominal.put(i, new HashSet[String]()) }
        if (!idxForNominal(i).contains(v + "") && !numIdx.contains(i)) { idxForNominal(i) += (v + "") }
      }
      data.+=(new Feature(j, target, features.map(f => f + ""), 1.0))
      j += 1
    })
  }
  
  def addInstances(inst: Feature) {
    data += (inst)
  }
  
//  def intWeight() {
//    var weightSum = 0.0
//    data.map(f => { weightSum += f.weight })
//    data.map(f => { f.weight = f.weight / weightSum })
//  }

//  def sample(factor: Double): Instances = {
//    val size = (data.size * factor).toInt
//    val insts = new Instances(numIdx)
//    for (i <- 0 to size) {
//      val v = (data.size * math.random).toInt
//      insts.addInstances(data.remove(v))
//    }
//    insts
//  }
  def copy(): Instances = {
    val insts = new Instances(numIdx)
    insts.labels = labels;
    insts.idxForNominal = idxForNominal;
    for (inst <- data.toIterator) {
      insts.addInstances(inst)
    }
    insts
  }
  //discretize the numerical feature
  def all2Normal(){
    val divs = Discretization.discretize(data, numIdx.toArray)
    divs.keys.map(i=>{
     
      var div = ArrayBuffer(Double.NegativeInfinity)
      div ++= divs(i)
      div += Double.PositiveInfinity
      
      data.map(f=>{
         val lf = f.features(i)
         if(!lf.equals("?")){
    	  val fi = f.features(i).toDouble
          var lastdiv:Double= Double.NegativeInfinity
          var target = "-1"
    	 for(t<-div){
          //var v = div.map(t=>{
            if(fi<t && target.equalsIgnoreCase("-1")){
              target = lastdiv+"=<f<"+t
            }
            lastdiv = t
          }
          f.features(i) = target
          idxForNominal(i) += (target)
      }else{
        idxForNominal(i) += (lf)
      }
      })
    })
    numIdx.clear
  }
  // slpit data sets into training sets and test sets
  def stratify(testsSetRate:Double=0.3):(Instances,Instances) = {
    val D = if(isRegression) data.groupBy(f=>"1") else data.groupBy(f=>f.target)
    
    var test = new Instances(numIdx,isRegression);
    test.labels = labels;
    test.idxForNominal = idxForNominal;
    
    var train = new Instances(numIdx,isRegression);
    train.labels = labels;
    train.idxForNominal = idxForNominal;
    
    D.map(f=>{
      //val target = f._1
      val dataForThistarget = f._2
      val testSize = (dataForThistarget.size * testsSetRate).toInt
      for(i<- 0 until testSize){
        val i = (dataForThistarget.size * math.random).toInt
        test.addInstances( dataForThistarget.remove(i))
      }
      dataForThistarget.map(f=>train.addInstances(f))
    })
    (train,test)
  }
  
}