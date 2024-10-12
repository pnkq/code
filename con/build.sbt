val sparkVersion = "3.1.2"
val bigdlVersion = "2.1.0"
val sparkNLPVersion = "4.3.2"


javacOptions ++= Seq("-encoding", "UTF-8")

lazy val commonSettings = Seq(  
  name := "con",
  organization := "phuonglh.com",
  version := "1.0.0",
  scalaVersion := "2.12.19",

  libraryDependencies ++= Seq(
    "org.apache.spark" %% "spark-core" % sparkVersion,
    "org.apache.spark" %% "spark-sql" % sparkVersion,
    "org.apache.spark" %% "spark-mllib" % sparkVersion,
    "com.intel.analytics.bigdl" % "bigdl-dllib-spark_3.1.2" % bigdlVersion,
    // "com.intel.analytics.bigdl.core.native.mkl" % "mkl-java-x86_64-linux" % "2.0.0", // BUG: java.lang.NoSuchMethodError: com.intel.analytics.bigdl.mkl.MKL.vsErf
    // "com.intel.analytics.bigdl.core.native.mkl" % "mkl-java-mac" % "2.0.0", // BUG: java.lang.NoSuchMethodError: com.intel.analytics.bigdl.mkl.MKL.vsErf
    "com.github.scopt" %% "scopt" % "4.1.0",
    "com.johnsnowlabs.nlp" %% "spark-nlp" % sparkNLPVersion,
    "org.openjfx" % "javafx" % "22" pomOnly(),
    "org.projectlombok" % "lombok" % "1.18.2" % "provided" // fix a bug wih Red Hat Fedora
  )
)

lazy val con = (project in file("."))
    .settings(commonSettings, 
      assembly / mainClass := Some("vlp.con.MED"),
      assembly / assemblyJarName := "con.jar",
      libraryDependencies ++= Seq(),
      run / fork := true
  )
