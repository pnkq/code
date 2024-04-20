ThisBuild / version := "2.0.0"

val sparkVersion = "3.1.2"
val bigdlVersion = "2.1.0"
val sparkNLPVersion = "4.3.2"

lazy val dsl = (project in file("."))
  .settings(
    name := "dsl",
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-core" % sparkVersion,
      "org.apache.spark" %% "spark-sql" % sparkVersion,
      "org.apache.spark" %% "spark-mllib" % sparkVersion,
      "com.intel.analytics.bigdl" % "bigdl-dllib-spark_3.1.2" % bigdlVersion,
      "com.intel.analytics.bigdl.core.native.mkl" % "mkl-java-x86_64-linux" % "2.0.0",
      "com.github.scopt" %% "scopt" % "4.1.0",
      "com.google.protobuf" % "protobuf-java" % "3.22.2",
      "org.scala-lang.modules" %% "scala-collection-compat" % "2.8.0",
      "com.johnsnowlabs.nlp" %% "spark-nlp" % sparkNLPVersion
    )
  )
