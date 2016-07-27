organization := "org.allenai"

name := "semparse"

version := "1.0"

scalaVersion := "2.11.7"

scalacOptions ++= Seq("-unchecked", "-deprecation", "-feature")

javacOptions += "-Xlint:unchecked"

fork := true

connectInput := true

cancelable in Global := true

javaOptions ++= Seq("-Xmx140g", "-Xms140g")

libraryDependencies ++= Seq(
  //"org.allenai.ari" %% "ari-controller" % "0.0.4-SNAPSHOT",
  "org.apache.spark" %% "spark-core" % "1.6.0",
  "org.elasticsearch" % "elasticsearch" % "2.3.4",
  "org.json4s" %% "json4s-native" % "3.2.11",
  "com.jayantkrish.jklol" % "jklol" % "1.1",
  "edu.cmu.ml.rtw" %%  "pra" % "3.4-SNAPSHOT",
  "edu.cmu.ml.rtw" %%  "matt-util" % "2.3.1",
  "edu.stanford.nlp" %  "stanford-corenlp" % "3.4.1",
  "edu.stanford.nlp" %  "stanford-corenlp" % "3.4.1" classifier "models",
  "org.scalatest" %% "scalatest" % "2.2.1" % "test",
  "com.typesafe.scala-logging" %% "scala-logging" % "3.4.0",
  "ch.qos.logback" %  "logback-classic" % "1.1.7"  // backend for scala-logging
).map(_.exclude("org.slf4j", "slf4j-log4j12"))

resolvers ++= Seq(
  Resolver.bintrayRepo("allenai", "private"),
  "AllenAI Releases" at "http://utility.allenai.org:8081/nexus/content/repositories/releases"
)

instrumentSettings
