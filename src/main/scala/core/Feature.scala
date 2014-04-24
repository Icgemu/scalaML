package core

/**
 * class represent a sample in input ,use in classification
 * 
 * @author Icgemu
 * @version 0.0.1
 */
class Feature(
    var i: Int  , //line index in the fi
    var target: String , //label for this lin
    var features: Array[String] , //attributes, exclude lab
    var weight: Double //initial  weight
    ) {

  /**
   * make a copy of this sample
   */
  def copy(): Feature = {
    new Feature(i, target, features.map(f => f), weight)
  }
}