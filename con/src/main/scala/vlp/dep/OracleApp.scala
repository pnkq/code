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
  
  def createSentence: Sentence = {
    Sentence(ListBuffer(
      Token("ROOT", mutable.Map(Label.Id -> "0", Label.Head -> "-1", Label.DependencyLabel -> "NA")),
      Token("Economic", mutable.Map(Label.Id -> "1", Label.Head -> "2", Label.DependencyLabel -> "nmod")),
      Token("news", mutable.Map(Label.Id -> "2", Label.Head -> "3", Label.DependencyLabel -> "subj")),
      Token("had", mutable.Map(Label.Id -> "3", Label.Head -> "0", Label.DependencyLabel -> "root")),
      Token("little", mutable.Map(Label.Id -> "4", Label.Head -> "5", Label.DependencyLabel -> "nmod")),
      Token("effect", mutable.Map(Label.Id -> "5", Label.Head -> "3", Label.DependencyLabel -> "nobj")),
      Token("on", mutable.Map(Label.Id -> "6", Label.Head -> "5", Label.DependencyLabel -> "nmod")),
      Token("financial", mutable.Map(Label.Id -> "7", Label.Head -> "8", Label.DependencyLabel -> "nmod")),
      Token("markets", mutable.Map(Label.Id -> "8", Label.Head -> "6", Label.DependencyLabel -> "pmod")),
      Token(".", mutable.Map(Label.Id -> "9", Label.Head -> "3", Label.DependencyLabel -> "punct"))
    ))
  }

  def featurize: Unit = {
    val sentence = createSentence
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

    // test 1
    val oracle = new OracleAS(new FeatureExtractor(false, false))
    // val graph = Graph(createSentence)
    // oracle.decode(graph).foreach(println)
    // println

    // // test 2
    val graphs = GraphReader.read("dat/dep/UD_English-EWT/en_ewt-ud-test.conllu")
    // decode the last graph (as shown in the end of this file)
    oracle.decode(graphs.last).foreach(println)

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

// # sent_id = reviews-211933-0003
// # text = He listens and is excellent in diagnosing, addressing and explaining the specific issues and suggesting exercises to use.
// 1	He	he	PRON	PRP	Case=Nom|Gender=Masc|Number=Sing|Person=3|PronType=Prs	2	nsubj	2:nsubj|5:nsubj	_
// 2	listens	listen	VERB	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	0	root	0:root	_
// 3	and	and	CCONJ	CC	_	5	cc	5:cc	_
// 4	is	be	AUX	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	5	cop	5:cop	_
// 5	excellent	excellent	ADJ	JJ	Degree=Pos	2	conj	2:conj:and	_
// 6	in	in	SCONJ	IN	_	7	mark	7:mark	_
// 7	diagnosing	diagnose	VERB	VBG	VerbForm=Ger	5	advcl	5:advcl:in	SpaceAfter=No
// 8	,	,	PUNCT	,	_	9	punct	9:punct	_
// 9	addressing	address	VERB	VBG	VerbForm=Ger	7	conj	5:advcl:in|7:conj:and	_
// 10	and	and	CCONJ	CC	_	11	cc	11:cc	_
// 11	explaining	explain	VERB	VBG	VerbForm=Ger	7	conj	5:advcl:in|7:conj:and	_
// 12	the	the	DET	DT	Definite=Def|PronType=Art	14	det	14:det	_
// 13	specific	specific	ADJ	JJ	Degree=Pos	14	amod	14:amod	_
// 14	issues	issue	NOUN	NNS	Number=Plur	7	obj	7:obj|9:obj|11:obj	_
// 15	and	and	CCONJ	CC	_	16	cc	16:cc	_
// 16	suggesting	suggest	VERB	VBG	VerbForm=Ger	7	conj	5:advcl:in|7:conj:and	_
// 17	exercises	exercise	NOUN	NNS	Number=Plur	16	obj	16:obj	_
// 18	to	to	PART	TO	_	19	mark	19:mark	_
// 19	use	use	VERB	VB	VerbForm=Inf	17	acl	17:acl:to	SpaceAfter=No
// 20	.	.	PUNCT	.	_	2	punct	2:punct	_

// # sent_id = newsgroup-groups.google.com_hiddennook_23708a8afef2f3a8_ENG_20041226_230600-0010
// # text = Listing his priorities, Abbas told supporters of the ruling Fatah party that he was determined to provide security to his people and continue the struggle against Israel's partially completed West Bank barrier.
// 1	Listing	list	VERB	VBG	VerbForm=Ger	6	advcl	6:advcl	_
// 2	his	his	PRON	PRP$	Case=Gen|Gender=Masc|Number=Sing|Person=3|Poss=Yes|PronType=Prs	3	nmod:poss	3:nmod:poss	_
// 3	priorities	priority	NOUN	NNS	Number=Plur	1	obj	1:obj	SpaceAfter=No
// 4	,	,	PUNCT	,	_	1	punct	1:punct	_
// 5	Abbas	Abbas	PROPN	NNP	Number=Sing	6	nsubj	6:nsubj	_
// 6	told	tell	VERB	VBD	Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin	0	root	0:root	_
// 7	supporters	supporter	NOUN	NNS	Number=Plur	6	iobj	6:iobj	_
// 8	of	of	ADP	IN	_	12	case	12:case	_
// 9	the	the	DET	DT	Definite=Def|PronType=Art	12	det	12:det	_
// 10	ruling	rule	VERB	VBG	VerbForm=Ger	12	amod	12:amod	_
// 11	Fatah	Fatah	PROPN	NNP	Number=Sing	12	compound	12:compound	_
// 12	party	party	NOUN	NN	Number=Sing	7	nmod	7:nmod:of	_
// 13	that	that	SCONJ	IN	_	16	mark	16:mark	_
// 14	he	he	PRON	PRP	Case=Nom|Gender=Masc|Number=Sing|Person=3|PronType=Prs	16	nsubj	16:nsubj|18:nsubj:xsubj|24:nsubj:xsubj	_
// 15	was	be	AUX	VBD	Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin	16	cop	16:cop	_
// 16	determined	determined	ADJ	JJ	Degree=Pos	6	ccomp	6:ccomp	_
// 17	to	to	PART	TO	_	18	mark	18:mark	_
// 18	provide	provide	VERB	VB	VerbForm=Inf	16	xcomp	16:xcomp	_
// 19	security	security	NOUN	NN	Number=Sing	18	obj	18:obj	_
// 20	to	to	ADP	IN	_	22	case	22:case	_
// 21	his	his	PRON	PRP$	Case=Gen|Gender=Masc|Number=Sing|Person=3|Poss=Yes|PronType=Prs	22	nmod:poss	22:nmod:poss	_
// 22	people	people	NOUN	NNS	Number=Plur	18	obl	18:obl:to	_
// 23	and	and	CCONJ	CC	_	24	cc	24:cc	_
// 24	continue	continue	VERB	VB	VerbForm=Inf	18	conj	16:xcomp|18:conj:and	_
// 25	the	the	DET	DT	Definite=Def|PronType=Art	26	det	26:det	_
// 26	struggle	struggle	NOUN	NN	Number=Sing	24	obj	24:obj	_
// 27	against	against	ADP	IN	_	34	case	34:case	_
// 28-29	Israel's	_	_	_	_	_	_	_	_
// 28	Israel	Israel	PROPN	NNP	Number=Sing	34	nmod:poss	34:nmod:poss	_
// 29	's	's	PART	POS	_	28	case	28:case	_
// 30	partially	partially	ADV	RB	_	31	advmod	31:advmod	_
// 31	completed	complete	VERB	VBN	Tense=Past|VerbForm=Part|Voice=Pass	34	amod	34:amod	_
// 32	West	West	PROPN	NNP	Number=Sing	33	compound	33:compound	_
// 33	Bank	Bank	PROPN	NNP	Number=Sing	34	compound	34:compound	_
// 34	barrier	barrier	NOUN	NN	Number=Sing	26	nmod	26:nmod:against	SpaceAfter=No
// 35	.	.	PUNCT	.	_	6	punct	6:punct	_
