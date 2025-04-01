val sparkVersion = "3.1.3"
val bigdlVersion = "2.4.0"
val sparkNLPVersion = "4.3.2"


javacOptions ++= Seq("-encoding", "UTF-8")

lazy val commonSettings = Seq(  
  name := "con",
  organization := "phuonglh.com",
  version := "1.0.0",
//  scalaVersion := "2.12.19",

  libraryDependencies ++= Seq(
    "org.apache.spark" %% "spark-core" % sparkVersion,
    "org.apache.spark" %% "spark-sql" % sparkVersion,
    "org.apache.spark" %% "spark-mllib" % sparkVersion,
    "com.intel.analytics.bigdl" % "bigdl-dllib-spark_3.1.3" % bigdlVersion,
    "com.intel.analytics.bigdl.core.native.math" % "bigdl-core-math-x86_64-linux" % "2.3.0",
    "com.intel.analytics.bigdl.core.native.math" % "bigdl-core-math-mac" % "2.3.0",
    "com.github.scopt" %% "scopt" % "4.1.0",
    "com.johnsnowlabs.nlp" %% "spark-nlp" % sparkNLPVersion,
  )
)

lazy val con = (project in file("."))
    .settings(commonSettings, 
      assembly / mainClass := Some("vlp.ner.NER"),
      assembly / assemblyJarName := "con.jar",
      libraryDependencies ++= Seq(),
      run / fork := true
  )
