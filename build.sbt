organization := "org.allenai"

name := "semparse"

scalaVersion := "2.11.7"

scalacOptions ++= Seq("-unchecked", "-deprecation")

javacOptions += "-Xlint:unchecked"

fork := true

connectInput := true

javaOptions ++= Seq("-Xmx160g")

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "1.6.0",
  "org.scalatest" % "scalatest_2.11" % "2.2.1" % "test",
  "org.json4s" %% "json4s-native" % "3.2.11",
  "com.jayantkrish.jklol" %% "jklol" % "1.1-SNAPSHOT",
  "edu.cmu.ml.rtw" %%  "matt-util" % "1.2.5"
)

instrumentSettings
