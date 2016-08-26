organization := "org.allenai"

name := "deep-learning-for-aristo"

version := "1.0"

scalaVersion := "2.11.7"

scalacOptions ++= Seq("-unchecked", "-deprecation", "-feature")

javacOptions += "-Xlint:unchecked"

fork := true

connectInput := true

cancelable in Global := true

javaOptions ++= Seq("-Xmx14g", "-Xms14g")

libraryDependencies ++= Seq(
  //"org.allenai.ari" %% "ari-controller" % "0.0.4-SNAPSHOT",
  "org.apache.spark" %% "spark-core" % "1.6.0",
  "org.elasticsearch" % "elasticsearch" % "2.3.4",
  "org.json4s" %% "json4s-native" % "3.2.11",
  "com.jayantkrish.jklol" % "jklol" % "1.1",
  "edu.cmu.ml.rtw" %%  "pra" % "3.4-SNAPSHOT",
  "edu.cmu.ml.rtw" %%  "matt-util" % "2.3.2",
  "edu.stanford.nlp" %  "stanford-corenlp" % "3.4.1",
  "edu.stanford.nlp" %  "stanford-corenlp" % "3.4.1" classifier "models",
  "org.scalatest" %% "scalatest" % "2.2.1" % "test",
  "com.typesafe.scala-logging" %% "scala-logging" % "3.4.0",
  "ch.qos.logback" %  "logback-classic" % "1.1.7"  // backend for scala-logging
).map(_.exclude("org.slf4j", "slf4j-log4j12"))

lazy val testPython = TaskKey[Unit]("testPython")

testPython := {
  val exitCode = { "py.test -v --cov=dlfa" ! }
  if (exitCode != 0) {
     error("Python tests failed")
  }
}

// TODO(matt): it'd be nicer if this would still execute scala tests if python tests fail...
(test in Test) <<= (test in Test) dependsOn (testPython)

instrumentSettings
