import org.apache.spark.ml.recommendation.ALS
//import org.apache.spark.ml.recommendation.ALS.Rating
import org.apache.spark.SparkConf
//import org.apache.spark.SparkContext
import org.apache.spark.sql.{SparkSession,Row}
import org.apache.spark.sql.types._
//object 单例对象
object Als {
  def main(args: Array[String]): Unit = {
    //val conf = new SparkConf().setMaster("local").setAppName("ALStest")
    //val sc = new SparkContext(conf)
    val myconf = new SparkConf()
    myconf.setMaster("local").setAppName("ALStest")
    val spark = SparkSession.builder.config(myconf).getOrCreate()

    //定义加载数据的schema
    val schema1 = StructType(Array(StructField("userId", LongType, true),
      StructField("movieId", LongType, true), StructField("rating", DoubleType, true)))

//    val schema1 = StructType(
//      StructField("userId",LongType)::
//      StructField("movieId",LongType)::
//      StructField("rating",DoubleType)::
//      Nil
//    )
    val df = spark.read.schema(schema1).format("csv").option("header","true")
      .load("C:\\Users\\HL_Wang\\Desktop\\data\\ratings.csv")

    //本身有时间戳这一列，去掉这一列
    val df1 = df.select("userId","movieId", "rating")
    //划分测试集和训练集合
    val Array(training, test) = df1.randomSplit(Array(0.8, 0.2))

    //val Array(training,test) = ratings
    val als = new ALS().setMaxIter(5).setRegParam(0.01).setUserCol("userId")
      .setItemCol("movieId").setUserCol("rating")
    val model =als.fit(training)
    val predictions = model.transform(test)
    predictions.show
    spark.stop()

  }

}
