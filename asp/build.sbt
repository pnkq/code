import Dependencies._

ThisBuild / scalaVersion     := "2.12.10"
ThisBuild / version          := "0.1.0"
ThisBuild / organization     := "pro.phuonglh"
ThisBuild / organizationName := "phuonglh"


val sparkVersion = "3.1.2"
val bigdlVersion = "2.1.0"

lazy val asp = (project in file("."))
  .settings(
    name := "asp",
    libraryDependencies ++= Seq(
      munit % Test,
      "org.apache.spark" %% "spark-core" % sparkVersion,
      "org.apache.spark" %% "spark-sql" % sparkVersion,
      "org.apache.spark" %% "spark-mllib" % sparkVersion,
      "com.intel.analytics.bigdl" % "bigdl-dllib-spark_3.1.2" % bigdlVersion,
      "com.intel.analytics.bigdl.core.native.math" % "bigdl-core-math-x86_64-linux" % "2.3.0",
      "com.intel.analytics.bigdl.core.native.math" % "bigdl-core-math-mac" % "2.3.0",
      "com.github.fommil.netlib" % "all" % "1.1.2" pomOnly(),
      "dev.ludovic.netlib" % "blas" % "3.0.5"
    )
  )
