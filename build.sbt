name := "com.dbmd-dev"
version := "0.1"
scalaVersion := "2.11.6"
// https://mvnrepository.com/artifact/org.apache.spark/spark-mllib

libraryDependencies ++= Seq("org.apache.spark" %% "spark-core" % "2.3.0",
                             "org.apache.spark" % "spark-sql_2.11" % "2.3.0",
                             "org.apache.spark" % "spark-mllib_2.11" % "2.3.0",
                             "org.scalanlp" %% "breeze" % "0.13.2",
                             "org.scalanlp" %% "breeze-natives" % "0.13.2",
                             "org.scalanlp" %% "breeze-viz" % "0.13.2",
                             "com.github.scopt" %% "scopt" % "4.0.0-RC2",
                             "com.github.fommil.netlib" % "all" % "1.1.2",
                             "org.clustering4ever" % "clustering4ever_2.11" % "0.9.4")
libraryDependencies += "com.typesafe.scala-logging" %% "scala-logging" % "3.9.2"
libraryDependencies += "org.scalactic" %% "scalactic" % "3.1.0"
libraryDependencies += "org.scalatest" %% "scalatest" % "3.1.0" % "test"
resolvers ++= Seq(
  "Sonatype Snapshots" at "http://oss.sonatype.org/content/repositories/snapshots",
  "Sonatype Releases" at "http://oss.sonatype.org/content/repositories/releases",
  "mvnrepository" at "http://mvnrepository.com/artifact/"
)
scalacOptions += "-feature"