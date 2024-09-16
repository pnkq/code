val scala3Version = "3.5.0"

lazy val root = project
  .in(file("."))
  .settings(
    name := "lcj",
    version := "1.0",

    scalaVersion := scala3Version,

    libraryDependencies += "org.scalameta" %% "munit" % "1.0.0" % Test,
    libraryDependencies += "dev.langchain4j" % "langchain4j-open-ai" % "0.34.0",
    libraryDependencies += "dev.langchain4j" % "langchain4j" % "0.34.0",
  )
