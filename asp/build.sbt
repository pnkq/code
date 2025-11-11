import Dependencies._

ThisBuild / scalaVersion     := "2.12.10"
ThisBuild / version          := "0.1.0"
ThisBuild / organization     := "pro.phuong"
ThisBuild / organizationName := "phuonglh"


val sparkVersion = "3.1.3"
val bigdlVersion = "2.4.0"

lazy val root = (project in file("."))
  .settings(
    name := "asp",
    fork := true, // important for JVM flags
    javaOptions ++= Seq(
      "--add-opens=java.base/java.nio=ALL-UNNAMED",
      "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
      "--add-opens=java.base/java.lang=ALL-UNNAMED"
    ),
    libraryDependencies ++= Seq(
      munit % Test,
      "org.apache.spark" %% "spark-core" % sparkVersion,
      "org.apache.spark" %% "spark-sql" % sparkVersion,
      "org.apache.spark" %% "spark-mllib" % sparkVersion,
      "com.intel.analytics.bigdl" % "bigdl-dllib-spark_3.1.3" % bigdlVersion,
      // "com.intel.analytics.bigdl" % "bigdl-dllib-spark_3.1.3" % bigdlVersion classifier "jar-with-dependencies",
      "com.intel.analytics.bigdl.core.native.math" % "bigdl-core-math-x86_64-linux" % "2.3.0",
      "com.intel.analytics.bigdl.core.native.math" % "bigdl-core-math-mac" % "2.3.0",
      "org.apache.logging.log4j" % "log4j-api" % "2.17.1",
      "org.apache.logging.log4j" % "log4j-core" % "2.17.1",
      "org.tensorflow" % "proto" % "1.15.0",
      "com.github.fommil.netlib" % "all" % "1.1.2" pomOnly(),
      "dev.ludovic.netlib" % "blas" % "3.0.5"
    )
  )
