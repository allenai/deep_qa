organization := "org.allenai"

name := "semparse"

version := "1.0"

scalaVersion := "2.11.7"

scalacOptions ++= Seq("-unchecked", "-deprecation")

javacOptions += "-Xlint:unchecked"

fork := true

connectInput := true

cancelable in Global := true

javaOptions ++= Seq("-Xmx4g", "-Xms4g")

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "1.6.0",
  "org.json4s" %% "json4s-native" % "3.2.11",
  "com.jayantkrish.jklol" % "jklol" % "1.1",
  "edu.cmu.ml.rtw" %%  "pra" % "3.3",
  "edu.cmu.ml.rtw" %%  "matt-util" % "2.2-SNAPSHOT",
  "edu.stanford.nlp" %  "stanford-corenlp" % "3.4.1",
  "edu.stanford.nlp" %  "stanford-corenlp" % "3.4.1" classifier "models",
  "org.scalatest" %% "scalatest" % "2.2.1" % "test"
)

instrumentSettings
