val scala3Version = "3.5.0"

lazy val lcj = project
  .in(file("."))
  .settings(
    name := "lcj",
    version := "1.0",

    scalaVersion := scala3Version,

    libraryDependencies ++= Seq(
      "org.scalameta" %% "munit" % "1.0.0" % Test,
      "dev.langchain4j" % "langchain4j-open-ai" % "0.35.0",
      "dev.langchain4j" % "langchain4j" % "0.35.0",
      "dev.langchain4j" % "langchain4j-hugging-face" % "0.35.0",
      "dev.langchain4j" % "langchain4j-easy-rag" % "0.35.0",
    )
  )
