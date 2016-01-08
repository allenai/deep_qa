organization := "org.allenai"

name := "semparse"

scalaVersion := "2.11.2"

scalacOptions ++= Seq("-unchecked", "-deprecation")

javacOptions += "-Xlint:unchecked"

fork := true

javaOptions ++= Seq("-Xmx160g")

libraryDependencies ++= Seq(
  "org.scalatest" % "scalatest_2.11" % "2.2.1" % "test",
  "org.json4s" %% "json4s-native" % "3.2.11",
  "edu.cmu.ml.rtw" %%  "matt-util" % "1.2"
)

instrumentSettings
