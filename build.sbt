organization := "org.allenai"

name := "deep-qa"

version := "0.1.5"

scalaVersion := "2.11.7"

scalacOptions ++= Seq("-unchecked", "-deprecation", "-feature")

javacOptions += "-Xlint:unchecked"

fork := true

connectInput := true

cancelable in Global := true

javaOptions ++= Seq("-Xmx15g", "-Xms15g")

// Required to work with / auto-compile with protocol buffers in Scala.
import com.trueaccord.scalapb.{ScalaPbPlugin => PB}

PB.protobufSettings

PB.runProtoc in PB.protobufConfig := (args =>
    com.github.os72.protocjar.Protoc.runProtoc("-v300" +: args.toArray))

version in PB.protobufConfig := "3.0.0-beta-2"

libraryDependencies ++= Seq(
  "org.apache.commons" % "commons-lang3" % "3.0",
  "org.apache.spark" %% "spark-core" % "1.6.0",
  "org.elasticsearch" % "elasticsearch" % "2.3.4",
  "com.opencsv" % "opencsv" % "3.8",
  "org.json4s" %% "json4s-native" % "3.2.11",
  "com.jayantkrish.jklol" % "jklol" % "1.1",
  "edu.cmu.ml.rtw" %%  "pra" % "3.4",
  "edu.cmu.ml.rtw" %%  "matt-util" % "2.3.2",
  "edu.stanford.nlp" %  "stanford-corenlp" % "3.4.1",
  "edu.stanford.nlp" %  "stanford-corenlp" % "3.4.1" classifier "models",
  "org.scalatest" %% "scalatest" % "2.2.1" % "test",
  "com.typesafe.scala-logging" %% "scala-logging" % "3.4.0",
  "ch.qos.logback" %  "logback-classic" % "1.1.7",  // backend for scala-logging
  "org.allenai.openie" %% "openie" % "4.2.4",

  // These are for running this as a solver with gRPC
  "io.grpc" % "grpc-okhttp" % "0.13.2",
  "com.trueaccord.scalapb" %% "scalapb-runtime-grpc" % (PB.scalapbVersion in PB.protobufConfig).value,
  "com.typesafe" % "config" % "1.2.1"
).map(_.exclude("org.slf4j", "slf4j-log4j12"))

// OpenIE depends on ClearNLP 2.0.2. The version of HPPC used in this version conflicts with
// ElasticSearch, so we need to override it with our updated private fork.
dependencyOverrides += "com.clearnlp" % "clearnlp" % "2.0.3-allenai"

instrumentSettings

licenses += ("Apache-2.0", url("https://www.apache.org/licenses/LICENSE-2.0"))

bintrayOrganization := Some("allenai")

bintrayRepository := "private"
