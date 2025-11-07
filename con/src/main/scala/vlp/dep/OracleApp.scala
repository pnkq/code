package vlp.dep

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import java.nio.file.Files
import java.nio.file.Paths
import java.nio.charset.Charset
import java.nio.file.OpenOption
import java.nio.file.StandardOpenOption
import org.json4s.jackson.Serialization


case class T(words: Seq[String], transitions: Seq[String])

/**
  * Created by phuonglh on 6/22/17.
  * 
  * This is a small app to test the [[Oracle]].
  * 
  */
object OracleApp {
  
  def createSentence1: Sentence = {
    Sentence(ListBuffer(
      Token("ROOT", mutable.Map(Label.Id -> "0", Label.Head -> "-1", Label.DependencyLabel -> "NA", Label.UniversalPartOfSpeech -> "ROOT")),
      Token("Economic", mutable.Map(Label.Id -> "1", Label.Head -> "2", Label.DependencyLabel -> "nmod", Label.UniversalPartOfSpeech -> "ADJ")),
      Token("news", mutable.Map(Label.Id -> "2", Label.Head -> "3", Label.DependencyLabel -> "subj", Label.UniversalPartOfSpeech -> "NN")),
      Token("had", mutable.Map(Label.Id -> "3", Label.Head -> "0", Label.DependencyLabel -> "root", Label.UniversalPartOfSpeech -> "VERB")),
      Token("little", mutable.Map(Label.Id -> "4", Label.Head -> "5", Label.DependencyLabel -> "nmod", Label.UniversalPartOfSpeech -> "ADJ")),
      Token("effect", mutable.Map(Label.Id -> "5", Label.Head -> "3", Label.DependencyLabel -> "nobj", Label.UniversalPartOfSpeech -> "NN")),
      Token("on", mutable.Map(Label.Id -> "6", Label.Head -> "5", Label.DependencyLabel -> "nmod", Label.UniversalPartOfSpeech -> "ADP")),
      Token("financial", mutable.Map(Label.Id -> "7", Label.Head -> "8", Label.DependencyLabel -> "nmod", Label.UniversalPartOfSpeech -> "ADJ")),
      Token("markets", mutable.Map(Label.Id -> "8", Label.Head -> "6", Label.DependencyLabel -> "pmod", Label.UniversalPartOfSpeech -> "NNS")),
      Token(".", mutable.Map(Label.Id -> "9", Label.Head -> "3", Label.DependencyLabel -> "punct", Label.UniversalPartOfSpeech -> "PUNCT"))
    ))
  }

  def createSentence2: Sentence = {
    Sentence(ListBuffer(
      Token("ROOT", mutable.Map(Label.Id -> "0", Label.Head -> "-1", Label.DependencyLabel -> "NA", Label.UniversalPartOfSpeech -> "ROOT")),
      Token("The", mutable.Map(Label.Id -> "1", Label.Head -> "2", Label.DependencyLabel -> "det", Label.UniversalPartOfSpeech -> "DET")),
      Token("case", mutable.Map(Label.Id -> "2", Label.Head -> "5", Label.DependencyLabel -> "subj", Label.UniversalPartOfSpeech -> "NOUN")),
      Token("against", mutable.Map(Label.Id -> "3", Label.Head -> "4", Label.DependencyLabel -> "case", Label.UniversalPartOfSpeech -> "ADP")),
      Token("Iran", mutable.Map(Label.Id -> "4", Label.Head -> "2", Label.DependencyLabel -> "nmod", Label.UniversalPartOfSpeech -> "PROPN")),
      Token("has", mutable.Map(Label.Id -> "5", Label.Head -> "0", Label.DependencyLabel -> "root", Label.UniversalPartOfSpeech -> "VERB")),
      Token("a", mutable.Map(Label.Id -> "6", Label.Head -> "7", Label.DependencyLabel -> "det", Label.UniversalPartOfSpeech -> "DET")),
      Token("feeling", mutable.Map(Label.Id -> "7", Label.Head -> "5", Label.DependencyLabel -> "obj", Label.UniversalPartOfSpeech -> "NOUN")),
      Token("of", mutable.Map(Label.Id -> "8", Label.Head -> "10", Label.DependencyLabel -> "case", Label.UniversalPartOfSpeech -> "ADP")),
      Token("Deja", mutable.Map(Label.Id -> "9", Label.Head -> "10", Label.DependencyLabel -> "compound", Label.UniversalPartOfSpeech -> "X")),
      Token("vu", mutable.Map(Label.Id -> "10", Label.Head -> "7", Label.DependencyLabel -> "nmod:of", Label.UniversalPartOfSpeech -> "X"))
    ))
  }

  def featurize: Unit = {
    val sentence = createSentence1
    println(sentence)

    val stack = new mutable.Stack[String]()
    val queue = new mutable.Queue[String]()
    sentence.tokens.foreach(token => queue.enqueue(token.id))
    val arcs = new ListBuffer[Dependency]()
    val config: Config = new ConfigAE(sentence, stack, queue, arcs).next("SH")
    println(config)

    val featureMap = Map[FeatureType.Value, Boolean](FeatureType.Word -> true, FeatureType.PartOfSpeech -> true)
    val extractor = new FeatureExtractor(false, false)
    val features = extractor.extract(config)
    println(features)
  }
  
  def run(pathCoNLLU: String, oracle: Oracle, pathOutput: String) = {
    val graphs = GraphReader.read(pathCoNLLU)
    import org.json4s._
    import org.json4s.jackson.Serialization    
    val lines = graphs.map { graph =>
      val configs = oracle.decode(graph)
      val words = graph.sentence.tokens.map(_.word)
      val transitions = configs.map(context => context.transition)
      Serialization.write(T(words, transitions))(org.json4s.DefaultFormats)
    }
    import scala.collection.JavaConverters._
    Files.write(Paths.get(pathOutput), lines.asJava, Charset.defaultCharset(), StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)
  }

  def main(args: Array[String]): Unit = {
    // test 0
    // featurize

    val oracle = new OracleAS(new FeatureExtractor(false, false))

    // test 1
    val graph = Graph(createSentence2)
    println(graph)
    oracle.decode(graph).foreach(println)
    println

    // // test 2
    // val graphs = GraphReader.read("dat/dep/UD_English-EWT/en_ewt-ud-test.conllu")
    // println(graphs.last)
    // // decode the last graph (as shown in the end of this file)
    // oracle.decode(graphs.last).foreach(println)
    // println

    // run("dat/dep/UD_English-EWT/en_ewt-ud-train.conllu", oracle, "dat/dep/en-as-train.jsonl")
    // run("dat/dep/UD_English-EWT/en_ewt-ud-dev.conllu", oracle, "dat/dep/en-as-dev.jsonl")
    // run("dat/dep/UD_English-EWT/en_ewt-ud-test.conllu", oracle, "dat/dep/en-as-test.jsonl")

    // val treebanks = Seq("atis", "eslspok", "ewt", "gum", "lines", "partut", "pud")
    // val dfs = treebanks.map { name =>
    //   val path = s"dat/dep/UD_English/$name"
    //   run(s"$path.conllu", oracle, s"$path.jsonl")
    // }

  }
}
