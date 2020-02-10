

object TestRunner {
    def main(args: Array[String]): Unit = {
        new TestMatUtil().execute()
        new TestMappers().execute()
    }
}
