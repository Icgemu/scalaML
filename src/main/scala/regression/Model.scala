package regression

import core.Instances

trait Model {

  def predict(test:Instances):Double
  
}