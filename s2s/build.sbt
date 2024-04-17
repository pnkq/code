ThisBuild / version := "1.0.0"

val sparkVersion = "3.1.2"
val bigdlVersion = "2.1.0"

resolvers += "Unidata" at "https://artifacts.unidata.ucar.edu/repository/unidata-all/"

lazy val s2s = (project in file("."))
  .settings(
    name := "s2s",
    libraryDependencies ++= Seq(
      "edu.ucar" % "cdm-core" % "5.5.2",
      "org.apache.spark" %% "spark-core" % sparkVersion,
      "org.apache.spark" %% "spark-sql" % sparkVersion,
      "org.apache.spark" %% "spark-mllib" % sparkVersion,
      "com.intel.analytics.bigdl" % "bigdl-dllib-spark_3.1.2" % bigdlVersion,
      "com.intel.analytics.bigdl.core.native.mkl" % "mkl-java-x86_64-linux" % "2.0.0",
      "com.intel.analytics.bigdl.core.native.mkl" % "mkl-java-mac" % "2.0.0",
      "com.github.scopt" %% "scopt" % "4.1.0",
      "com.google.protobuf" % "protobuf-java" % "3.22.2",
      "org.scala-lang.modules" %% "scala-collection-compat" % "2.8.0",
      "io.github.cibotech" %% "evilplot" % "0.9.0",
      "io.github.cibotech" %% "evilplot-repl" % "0.9.0"
    )
  )
