addSbtPlugin("de.johoop" % "jacoco4sbt" % "2.1.6")

addSbtPlugin("com.trueaccord.scalapb" % "sbt-scalapb" % "0.5.26")

libraryDependencies ++= Seq(
  "com.github.os72" % "protoc-jar" % "3.0.0-b2.1"
)

addSbtPlugin("net.virtual-void" % "sbt-dependency-graph" % "0.8.2")
