ThisBuild / version := "1.0.0"

val sparkVersion = "3.1.3"

resolvers += "Unidata" at "https://artifacts.unidata.ucar.edu/repository/unidata-all/"

lazy val s2s = (project in file("."))
  .settings(
    name := "s2s",
    libraryDependencies ++= Seq(
      "edu.ucar" % "cdm-core" % "5.5.2",
      "org.apache.spark" %% "spark-core" % sparkVersion,
      "org.apache.spark" %% "spark-sql" % sparkVersion,
      "org.apache.spark" %% "spark-mllib" % sparkVersion,
      "com.intel.analytics.bigdl" % "bigdl-dllib-spark_3.1.3" % "2.2.0",
      "com.github.scopt" %% "scopt" % "4.1.0",
      "io.github.cibotech" %% "evilplot" % "0.9.0",
      "io.github.cibotech" %% "evilplot-repl" % "0.9.0",
      "com.lihaoyi" %% "upickle" % "3.3.0"
    )
  )
