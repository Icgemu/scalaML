package algorithm.ann

import org.apache.spark.mllib.regression.LabeledPoint
import scala.io.Source
import java.io.File
import scala.Array.canBuildFrom

object NeuralNetworks {
  
  //x data has 1 in index 0
  def forwordProp(x:Array[Double],y:Double,theta:Array[Array[Array[Double]]],
      gama :Array[Array[Double]],
      a_forLayer:Array[Array[Double]],bias_forLayer:Array[Array[Double]]) = {
    
    a_forLayer(0) = x
    
    for(i <-1 until a_forLayer.length){
      val l = a_forLayer(i)
      var j = 0
      val notlast = i<(a_forLayer.length -1)
      a_forLayer(i)=l.map(f=>{
        val result = if (j==0 && notlast) 1.0 else {
          
          var v = theta(i)(j).zip(a_forLayer(i-1))
          val r = v.map(p=>{
            p._1*p._2
          }).reduce(_ + _)
          //if(notlast==true){
        	  1/(1+math.exp(-r-bias_forLayer(i)(j)))
//          }else{
//            r
//          }
        }
        j = j+1
        result
      })
    }
    
    backfordProp(y,theta,gama,a_forLayer,bias_forLayer)
  }
  
  def backfordProp(y:Double,theta:Array[Array[Array[Double]]],
      gama :Array[Array[Double]],
      a_forLayer:Array[Array[Double]],bias_forLayer:Array[Array[Double]]) = {
      val last = a_forLayer.last
      var no = 0
      gama(gama.length-1) = last.map(f=>{
       var ret = 0.0
       if(y==no+1){ 
         ret = (1-f)*(f*(1-f))
       } else{
         ret = (0-f)*(f*(1-f))
       }
//       if(y==no+1){ 
//         ret = (1-f)
//       } else{
//         ret = (0-f)
//       }
        //ret = (y-f)
       bias_forLayer(bias_forLayer.length-1)(no)=bias_forLayer(bias_forLayer.length-1)(no) + 0.01*ret
       no = no + 1
       
       ret
      })
      //gama 
      for(i<-1 until gama.length-1){
        val j=gama.length-1 -i
        val l = gama(j).length;
        val l1 = gama(j+1).length;
        
        for(k<-1 until l){
          gama(j)(k) = {
            var sum =0.0
            val out = if(i==1) 0 else 1
            for(t <-out until l1){
              sum+=theta(j+1)(t)(k)*gama(j+1)(t)
              //sum+=theta(j+1)(t)(k)*gama(j+1)(t)*a_forLayer(j+1)(t)*(1-a_forLayer(j+1)(t))
            }
            //val err = if(i==1)sum else sum*a_forLayer(j)(k)*(1-a_forLayer(j)(k))
             val err = sum*a_forLayer(j)(k)*(1-a_forLayer(j)(k))
            bias_forLayer(j)(k) = bias_forLayer(j)(k) + 0.01*err
            err
          }
        }
      }
      //renew theta
      
      for(i <- 1 until theta.length){
        //theta(i)={
          var l = theta(i)
          val out = if(i== theta.length-1) 0 else 1
          for(j<-out until l.length){
            val t = theta(i)(j)
            //val out = if(i== theta.length-1) 0 else 1
            for(k<-0 until t.length){
              theta(i)(j)(k) = theta(i)(j)(k)+0.01*(gama(i)(j)*a_forLayer(i-1)(k))
            }
          //}
          
          //theta(i)
        }
      }
  }
  
  def predict(x:Array[Double],theta:Array[Array[Array[Double]]],
      gama :Array[Array[Double]],
      a_forLayer:Array[Array[Double]],bias_forLayer:Array[Array[Double]]) = {
    
    a_forLayer(0) = x
    
    for(i <-1 until a_forLayer.length){
      val l = a_forLayer(i)
      var j = 0
      val notlast = i<(a_forLayer.length -1)
      a_forLayer(i)=l.map(f=>{
        val result = if (j==0 && notlast) 1.0 else {
          
          var v = theta(i)(j).zip(a_forLayer(i-1))
          val r = v.map(p=>{
            p._1*p._2
          }).reduce(_ + _)
          //if(notlast==true){
        	  1/(1+math.exp(-r-bias_forLayer(i)(j)))
//          }else{
//            r
//          }
        }
        j = j+1
        result
      })
    }
    
    var last = a_forLayer.last
    //last=last.drop(0)
    for(i<-last){
      System.out.print(i+" ")
    }
    System.out.println(" ")
    //backfordProp(y,theta,gama,a_forLayer)
  }

  def loadCsv():Array[LabeledPoint]={
    val f = Source.fromFile(new File("D://tmp/iris.csv"))
    val x = f.getLines.toArray.map(line=>{
      val l = line.split(":")
      val data = l(0).split(",").map(v=>v.toDouble)
      LabeledPoint(l(1).toDouble,Array(1,data:_*))
    })
    x
  }
  
  def main(args: Array[String]): Unit = {
    
    //val layers = 5
    val unitNum_forLayer = Array(4,8,3)
    
    //theta layer to layer forword
    var theta_forLayer = unitNum_forLayer.map(f=>{
      val v = new Array[Array[Double]](f)
      v
     })
     var i = 0 ;
     for(k <- 0 until (theta_forLayer.length-1) ){
       
       val a = theta_forLayer(k)
       val b = theta_forLayer(k+1)
       if(k<theta_forLayer.length -2){
	       for(j <- 1 until b.length){
	           theta_forLayer(k+1)(j) = new Array[Double](a.length)
	           for(i <- 0 until a.length){
	        	   theta_forLayer(k+1)(j)(i) = math.random-0.5
	           }
	       }
       }else{
         
         for(j <- 0 until b.length){
	           theta_forLayer(k+1)(j) = new Array[Double](a.length)
	           for(i <- 0 until a.length){
	        	   theta_forLayer(k+1)(j)(i) = math.random-0.5
	           }
	       }
       }
       
     }
     //output of layer
     var a_forLayer = unitNum_forLayer.map(f=>{
      val v = Array.fill(f)(1.0)
      v
     })
     //output of layer
     var bias_forLayer = unitNum_forLayer.map(f=>{
      val v = Array.fill(f)(1.0)
      val t = v.map(f=>math.random)
      t
     })
     //error
     var gama_forLayer = unitNum_forLayer.map(f=>{
      val v = Array.fill(f)(1.0)
      val t = v.map(f=>math.random)
      t
     })
     
//     var x = Array[LabeledPoint](
//      LabeledPoint(1, Array(1,(47-34)*1.0/(56-34), (76-74)*1.0/(77-74), (24-21)*1.0/(25-21))),
//      LabeledPoint(1, Array(1,(46-34)*1.0/(56-34), (77-74)*1.0/(77-74), (23-21)*1.0/(25-21))),
//      LabeledPoint(1, Array(1,(48-34)*1.0/(56-34), (74-74)*1.0/(77-74), (22-21)*1.0/(25-21))),
//      LabeledPoint(2, Array(1,(34-34)*1.0/(56-34), (76-74)*1.0/(77-74), (21-21)*1.0/(25-21))),
//      LabeledPoint(2, Array(1,(35-34)*1.0/(56-34), (75-74)*1.0/(77-74), (24-21)*1.0/(25-21))),
//      LabeledPoint(2, Array(1,(34-34)*1.0/(56-34), (77-74)*1.0/(77-74), (25-21)*1.0/(25-21))),
//      LabeledPoint(3, Array(1,(55-34)*1.0/(56-34), (76-74)*1.0/(77-74), (21-21)*1.0/(25-21))),
//      LabeledPoint(3, Array(1,(54-34)*1.0/(56-34), (75-74)*1.0/(77-74), (23-21)*1.0/(25-21))),
//      LabeledPoint(3, Array(1,(56-34)*1.0/(56-34), (77-74)*1.0/(77-74), (25-21)*1.0/(25-21)))
//     )
          var x = Array[LabeledPoint](
      LabeledPoint(2, Array(1, 31, 76, 21)),
      LabeledPoint(2, Array(1, 32, 75, 24)),
      LabeledPoint(2, Array(1, 30, 77, 25)),
      LabeledPoint(1, Array(1, 47, 76, 24)),
      LabeledPoint(1, Array(1, 46, 77, 23)),
      LabeledPoint(1, Array(1, 48, 74, 22)),
      LabeledPoint(3, Array(1, 55, 76, 21)),
      LabeledPoint(3, Array(1, 56, 74, 22)),
      LabeledPoint(3,  Array(1, 55, 72, 22))
     )
//     var x = loadCsv()
     for(i<-0 to 1000){
       for(xi<-x){
         forwordProp(xi.features,xi.label,theta_forLayer,gama_forLayer,a_forLayer,bias_forLayer)
       }
     }
     
     for(xi<-x){
         predict(xi.features,theta_forLayer,gama_forLayer,a_forLayer,bias_forLayer)
     }
     //predict(Array(1,58, 76, 28),theta_forLayer,gama_forLayer,a_forLayer,bias_forLayer)
     //predict(Array(1,30, 76, 23),theta_forLayer,gama_forLayer,a_forLayer,bias_forLayer)
//predict(Array(1,5.7,4.4,1.5,0.4),theta_forLayer,gama_forLayer,a_forLayer,bias_forLayer)
//predict(Array(1,6.4,3.2,4.5,1.5),theta_forLayer,gama_forLayer,a_forLayer,bias_forLayer)
//predict(Array(1,6.4,2.7,5.3,1.9),theta_forLayer,gama_forLayer,a_forLayer,bias_forLayer)
     //5.7,4.4,1.5,0.4:1
     //6.4,3.2,4.5,1.5:2
     //6.4,2.7,5.3,1.9:3
  }

}