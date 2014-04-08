package algorithm

class LabeledFeature(var i:Int,var label:String,
    var features:Array[String],
    var weight:Double) {
  
  
  def copy():LabeledFeature={
    new LabeledFeature(i,label,features.map(f=>f),weight)
  }
}