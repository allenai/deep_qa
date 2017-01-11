package org.allenai.deep_qa.data

import com.mattg.util.FakeFileUtil
import org.scalatest._

class SquadDatasetReaderSpec extends FlatSpecLike with Matchers {

  val fileUtil = new FakeFileUtil
  val paragraph1 = "Architecturally, the school has a Catholic character. Atop the Main " +
    "Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main " +
    "Building and facing it, is a copper statue of Christ with arms upraised with the legend " +
    "\\\"Venite Ad Me Omnes\\\". Next to the Main Building is the Basilica of the Sacred Heart. " +
    "Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is " +
    "a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint " +
    "Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that " +
    "connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary."
  val question11 = "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?"
  val answer11 = "Saint Bernadette Soubirous"
  val question12 = "What sits on top of the Main Building at Notre Dame?"
  val answer12 = "a golden statue of the Virgin Mary"
  val paragraph2 = "In 1882, Albert Zahm (John Zahm's brother) built an early wind tunnel used " +
    "to compare lift to drag of aeronautical models. Around 1899, Professor Jerome Green became " +
    "the first American to send a wireless message. In 1931, Father Julius Nieuwland performed " +
    "early work on basic reactions that was used to create neoprene. Study of nuclear physics at " +
    "the university began with the building of a nuclear accelerator in 1936, and continues now " +
    "partly through a partnership in the Joint Institute for Nuclear Astrophysics."
  val question21 = "In what year did Albert Zahm begin comparing aeronatical models at Notre Dame?"
  val answer21 = "1882"
  val question22 = "Which individual worked on projects at Notre Dame that eventually created neoprene?"
  val answer22 = "Father Julius Nieuwland"
  val question23 = "What did the brother of John Zahm construct at Notre Dame?"
  val answer23 = "an early wind tunnel"
  val paragraph3 = "The iPod is a line of portable media players and multi-purpose pocket " +
    "computers designed and marketed by Apple Inc. The first line was released on October 23, " +
    "2001, about 8\u00bd months after iTunes (Macintosh version) was released. The most recent " +
    "iPod redesigns were announced on July 15, 2015. There are three current versions of the " +
    "iPod: the ultra-compact iPod Shuffle, the compact iPod Nano and the touchscreen iPod Touch."
  val question31 = "Which company produces the iPod?"
  val answer31 = "Apple"
  val question32 = "In what year was the iPod most recently redesigned?"
  val answer32 = "2015"
  val datasetFile = "/dataset"
  val datasetFileContents = s"""{
      |"data": [
      |  {
      |    "title": "University_of_Notre_Dame",
      |    "paragraphs": [
      |          {
      |              "context": "${paragraph1}",
      |              "qas": [
      |                  {
      |                      "answers": [
      |                          {
      |                              "answer_start": 515,
      |                              "text": "${answer11}"
      |                          }
      |                      ],
      |                      "question": "${question11}",
      |                      "id": "5733be284776f41900661182"
      |                  },
      |                  {
      |                      "answers": [
      |                          {
      |                              "answer_start": 92,
      |                              "text": "${answer12}"
      |                          }
      |                      ],
      |                      "question": "${question12}",
      |                      "id": "5733be284776f4190066117e"
      |                  }
      |              ]
      |          },
      |          {
      |              "context": "${paragraph2}",
      |              "qas": [
      |                  {
      |                      "answers": [
      |                          {
      |                              "answer_start": 3,
      |                              "text": "${answer21}"
      |                          }
      |                      ],
      |                      "question": "${question21}",
      |                      "id": "5733b1da4776f41900661068"
      |                  },
      |                  {
      |                      "answers": [
      |                          {
      |                              "answer_start": 222,
      |                              "text": "${answer22}"
      |                          }
      |                      ],
      |                      "question": "${question22}",
      |                      "id": "5733b1da4776f4190066106b"
      |                  },
      |                  {
      |                      "answers": [
      |                          {
      |                              "answer_start": 49,
      |                              "text": "${answer23}"
      |                          }
      |                      ],
      |                      "question": "${question23}",
      |                      "id": "5733b1da4776f41900661067"
      |                  }
      |              ]
      |          },
      |    ]
      |  },
      |  {
      |      "title": "IPod",
      |      "paragraphs": [
      |          {
      |              "context": "${paragraph3}",
      |              "qas": [
      |                  {
      |                      "answers": [
      |                          {
      |                              "answer_start": 105,
      |                              "text": "${answer31}"
      |                          }
      |                      ],
      |                      "question": "${question31}",
      |                      "id": "56cc55856d243a140015ef0a"
      |                  },
      |                  {
      |                      "answers": [
      |                          {
      |                              "answer_start": 286,
      |                              "text": "${answer32}"
      |                          }
      |                      ],
      |                      "question": "${question32}",
      |                      "id": "56ce726faab44d1400b88795"
      |                  }
      |              ]
      |          },
      |       ]
      |  }
      |],
      |"version": "1.1"
      |}""".stripMargin

  fileUtil.addFileToBeRead(datasetFile, datasetFileContents)
  val reader = new SquadDatasetReader(fileUtil)

  "readFile" should "return a correct dataset" in {
    val dataset = reader.readFile(datasetFile)

    val fixedParagraph1 = paragraph1.replace("\\\"", "\"")
    dataset.instances.size should be(7)
    dataset.instances(0) should be(SpanPredictionInstance(question11, fixedParagraph1, Some(515, 515 + answer11.size)))
    dataset.instances(1) should be(SpanPredictionInstance(question12, fixedParagraph1, Some(92, 92 + answer12.size)))
    dataset.instances(2) should be(SpanPredictionInstance(question21, paragraph2, Some(3, 3 + answer21.size)))
    dataset.instances(3) should be(SpanPredictionInstance(question22, paragraph2, Some(222, 222 + answer22.size)))
    dataset.instances(4) should be(SpanPredictionInstance(question23, paragraph2, Some(49, 49 + answer23.size)))
    dataset.instances(5) should be(SpanPredictionInstance(question31, paragraph3, Some(105, 105 + answer31.size)))
    dataset.instances(6) should be(SpanPredictionInstance(question32, paragraph3, Some(286, 286 + answer32.size)))
  }
}

