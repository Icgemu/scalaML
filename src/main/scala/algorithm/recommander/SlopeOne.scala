package algorithm.recommander
import scala.collection.mutable.HashMap

object SlopeOne {

  var mData = new HashMap[Int, HashMap[Int, Double]]()
  var mDiffMatrix = new HashMap[Int, HashMap[Int, Double]]()
  var mFreqMatrix = new HashMap[Int, HashMap[Int, Int]]()

  var mAllItems = Array[Int]()

  def buildit(file: String) {
    //    val buff = Source.fromFile(new File(file))
    //    buff.getLines.toArray.map(line => {
    //      val arr = line.split("	")
    //      val iid = arr(1).toInt - 1
    //      val uid = arr(0).toInt - 1
    //      val sc = arr(2).toDouble
    //      if (!mData.contains(uid)) {
    //        mData(uid) = new HashMap[Int, Double]()
    //      }
    //      mData(uid) += ((iid, sc))
    //    })
    //    buff.close
    mData.map(user => {
      val user2 = user._2
      val user3 = user._2
      println("User" + user._1)
      user2.map(entry => {
        val item = entry._1
        val score = entry._2
        if (!mDiffMatrix.contains(item)) {
          mDiffMatrix(item) = new HashMap[Int, Double]()
          mFreqMatrix(item) = new HashMap[Int, Int]()
        }

        user3.map(entry2 => {
          val item2 = entry2._1
          val score2 = entry2._2

          var oldcount = 0;
          if (mFreqMatrix(item).contains(item2))
            oldcount = mFreqMatrix(item)(item2)
          var olddiff = 0.0
          if (mDiffMatrix(item).contains(item2))
            olddiff = mDiffMatrix(item)(item2)
          val observeddiff = score - score2
          mFreqMatrix(item).put(item2, oldcount + 1)
          mDiffMatrix(item).put(item2, olddiff + observeddiff)
        })
      })
    })

    mDiffMatrix.map(item => {
      val j = item._1
      item._2.map(t => {
        val i = t._1
        val oldvalue = mDiffMatrix(j)(i)
        val count = mFreqMatrix(j)(i)
        mDiffMatrix(j).put(i, oldvalue / count)
      })
    })
  }

  def predict(user: HashMap[Int, Double]): HashMap[Int, Double] = {
    var predictions = new HashMap[Int, Double]()
    var frequencies = new HashMap[Int, Int]()
    for (j <- mDiffMatrix.keys) {
      frequencies.put(j, 0);
      predictions.put(j, 0.0);
    }
    for (j <- user.keys) {
      for (k <- mDiffMatrix.keys) {
        try {
          val newval = (mDiffMatrix(k)(j) + user(j)) * mFreqMatrix(k)(j)
          predictions.put(k, predictions(k) + newval)
          frequencies.put(k, frequencies(k) + mFreqMatrix(k)(j))
        } catch {
          case ex: Exception => {}
        }
      }
    }
    var cleanpredictions = new HashMap[Int, Double]()
    for (j <- predictions.keys) {
      if (frequencies(j) > 0) {
        cleanpredictions.put(j, predictions(j) / frequencies(j))
      }
    }
    for (j <- user.keys) {
      cleanpredictions.put(j, user(j))
    }
    cleanpredictions
  }

  def weightlesspredict(user: HashMap[Int, Double]): HashMap[Int, Double] = {
    var predictions = new HashMap[Int, Double]()
    var frequencies = new HashMap[Int, Int]()
    for (j <- mDiffMatrix.keys) {
      frequencies.put(j, 0);
      predictions.put(j, 0.0);
    }
    for (j <- user.keys) {
      for (k <- mDiffMatrix.keys) {
        //try {
        val newval = (mDiffMatrix(k)(j) + user(j)) //* mFreqMatrix(k)(j)
        predictions.put(k, predictions(k) + newval)
        //frequencies.put(k, frequencies(k)+mFreqMatrix(k)(j))
        //} catch(NullPointerException e) {}
      }
    }
    for (j <- predictions.keys) {
      predictions.put(j, predictions(j) / user.size)
    }
    for (j <- user.keys) {
      predictions.put(j, user(j));
    }
    predictions
  }
  def printData() {
    for (user <- mData.keys) {
      System.out.println(user);
      print(mData(user));
    }
    for (i <- 0 until mAllItems.length) {
      System.out.print("\n" + mAllItems(i) + ":");
      printMatrixes(mDiffMatrix(mAllItems(i)), mFreqMatrix(mAllItems(i)));
    }
  }

  def printMatrixes(ratings: HashMap[Int, Double],
    frequencies: HashMap[Int, Int]) {
    for (j <- 0 until mAllItems.length) {
      System.out.format("%10.3f", ratings.get(mAllItems(j)));
      System.out.print(" ");
      System.out.format("%10d", frequencies.get(mAllItems(j)));
    }
    System.out.println();
  }
  def print(user: HashMap[Int, Double]) {
    for (j <- user.keys) {
      System.out.println(" " + j + " --> " + user(j))
    }
  }

  def main(args: Array[String]): Unit = {

    var data = new HashMap[Int, HashMap[Int, Double]]();
    // items
    val item1 = 1
    val item2 = 2
    val item3 = 3
    val item4 = 4
    val item5 = 5

    mAllItems = Array(item1, item2, item3, item4, item5)

    //I'm going to fill it in
    var user1 = new HashMap[Int, Double]();
    var user2 = new HashMap[Int, Double]();
    var user3 = new HashMap[Int, Double]();
    var user4 = new HashMap[Int, Double]();
    user1.put(item1, 1.0f);
    user1.put(item2, 0.5f);
    user1.put(item4, 0.1f);
    data.put(11, user1);
    user2.put(item1, 1.0f);
    user2.put(item3, 0.5f);
    user2.put(item4, 0.2f);
    data.put(12, user2);
    user3.put(item1, 0.9f);
    user3.put(item2, 0.4f);
    user3.put(item3, 0.5f);
    user3.put(item4, 0.1f);
    data.put(13, user3);
    user4.put(item1, 0.1f);
    //user4.put(item2,0.4f);
    //user4.put(item3,0.5f);
    user4.put(item4, 1.0f);
    user4.put(item5, 0.4f);
    data.put(14, user4);
    mData = data
    buildit("E://books/spark/ml/svd/SVD_/ml_data/training.txt")

    // then, I'm going to test it out...
    var user = new HashMap[Int, Double]();
    System.out.println("Ok, now we predict...");
    user.put(item5, 0.4f);
    System.out.println("Inputting...");
    SlopeOne.print(user);
    System.out.println("Getting...");
    predict(user)
    SlopeOne.print(predict(user));
    //
    user.put(item4, 0.2f);
    System.out.println("Inputting...");
    SlopeOne.print(user);
    System.out.println("Getting...");
    SlopeOne.print(predict(user));
  }

}