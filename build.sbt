organization := "org.allenai"

name := "semparse"

scalaVersion := "2.10.5"

scalacOptions ++= Seq("-unchecked", "-deprecation")

javacOptions += "-Xlint:unchecked"

fork := true

connectInput := true

javaOptions ++= Seq("-Xmx140g", "-Xms140g")

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "1.6.0",
  "org.json4s" %% "json4s-native" % "3.2.11",
  "com.jayantkrish.jklol" %% "jklol" % "1.1-SNAPSHOT",
  "edu.cmu.ml.rtw" %%  "pra" % "3.2.1-SNAPSHOT",
  "edu.cmu.ml.rtw" %%  "matt-util" % "2.1-SNAPSHOT",
  "org.scalatest" %% "scalatest" % "2.2.1" % "test"
)

instrumentSettings
