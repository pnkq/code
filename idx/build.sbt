ThisBuild / version := "1.0.0"


lazy val idx = (project in file("."))
  .settings(
    name := "idx",
    libraryDependencies ++= Seq(
      "org.elasticsearch.client" % "elasticsearch-rest-high-level-client" % "7.1.1",
      "de.l3s.boilerpipe" % "boilerpipe" % "1.1.0",
      "xerces" % "xercesImpl" % "2.11.0",
      "net.sourceforge.nekohtml" % "nekohtml" % "1.9.22" % "provided",
      "org.json4s" %% "json4s-jackson" % "4.0.0",
      "org.apache.logging.log4j" % "log4j-core" % "2.17.1",
      "commons-io" % "commons-io" % "2.13.0",
      "org.apache.kafka" % "kafka-clients" % "2.6.0"
    )
  )
