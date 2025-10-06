package vlp.dep

import scala.collection.mutable
import scala.collection.mutable.ListBuffer

/**
  * Created by phuonglh on 6/22/17.
  * 
  * This is a small app to test the [[Oracle]].
  * 
  */
object OracleApp {
  
  def createSentence: Sentence = {
    val m0 = mutable.Map[Label.Value, String](Label.Id -> "0", Label.PartOfSpeech -> "ROOT", Label.Head -> "-1")
    val t0 = Token("ROOT", m0)
    val m1 = mutable.Map[Label.Value, String](Label.Id -> "1", Label.PartOfSpeech -> "PRON", Label.Head -> "2")
    val t1 = Token("I", m1)
    val m2 = mutable.Map[Label.Value, String](Label.Id -> "2", Label.PartOfSpeech -> "VERB", Label.Head -> "0")
    val t2 = Token("love", m2)
    val m3 = mutable.Map[Label.Value, String](Label.Id -> "3", Label.PartOfSpeech -> "PRON", Label.Head -> "2")
    val t3 = Token("you", m3)
    Sentence(ListBuffer(t0, t1, t2, t3))
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
  
  def main(args: Array[String]): Unit = {
    // test 1
    // featurize

    // test 2
    val graphs = GraphReader.read("dat/dep/UD_English-EWT/en_ewt-ud-test.conllu")
    val oracle = new OracleAE(new FeatureExtractor(false, false))
    // decode the last graph (as shown in the end of this file)
    val contexts = oracle.decode(graphs.last)
    contexts.foreach(println)
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

