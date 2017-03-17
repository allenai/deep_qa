# pylint: disable=no-self-use,invalid-name
from os.path import join

from overrides import overrides
from deep_qa.data.dataset_readers.squad_sentence_selection_reader import SquadSentenceSelectionReader
from ...common.test_case import DeepQaTestCase


class TestSquadSentenceSelectionReader(DeepQaTestCase):
    @overrides
    def setUp(self):
        super(TestSquadSentenceSelectionReader, self).setUp()
        # write a SQuAD json file.
        passage = ("Architecturally, the school has a Catholic character. Atop the "
                   "Main Building's gold dome is a golden statue of the Virgin Mary. "
                   "Immediately in front of the Main Building and facing it, is a copper "
                   "statue of Christ with arms upraised with the legend \\\"Venite Ad Me Omnes\\\". "
                   "Next to the Main Building is the Basilica of the Sacred Heart. Immediately "
                   "behind the basilica is the Grotto, a Marian place of prayer and reflection. "
                   "It is a replica of the grotto at Lourdes, France where the Virgin Mary "
                   "reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the "
                   "main drive (and in a direct line that connects through 3 statues and the Gold "
                   "Dome), is a simple, modern stone statue of Mary.")
        self.question0 = "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?"
        self.question1 = "What is in front of the Notre Dame Main Building?"
        json_string = """
        {
          "data":[
            {
              "title":"University_of_Notre_Dame",
              "paragraphs":[
                {
                  "context":"%s",
                  "qas":[
                    {
                      "answers":[
                        {
                          "answer_start":515,
                          "text":"Saint Bernadette Soubirous"
                        }
                      ],
                      "question":"%s",
                      "id":"5733be284776f41900661182"
                    },
                    {
                      "answers":[
                        {
                          "answer_start":188,
                          "text":"a copper statue of Christ"
                        }
                      ],
                      "question":"%s",
                      "id":"5733be284776f4190066117f"
                    }
                  ]
                }
              ]
            }
          ]
        }
        """ % (passage, self.question0, self.question1)
        with open(self.TEST_DIR + "squad_data.json", "w") as f:
            f.write(json_string)

    def test_default_squad_sentence_selection_reader(self):
        context0 = ("Architecturally, the school has a Catholic character.###Atop "
                    "the Main Building's gold dome is a golden statue of the Virgin "
                    "Mary.###Immediately behind the basilica is the Grotto, a "
                    "Marian place of prayer and reflection.###Next to the Main "
                    "Building is the Basilica of the Sacred Heart.###Immediately "
                    "in front of the Main Building and facing it, is a copper "
                    "statue of Christ with arms upraised with the legend \"Venite "
                    "Ad Me Omnes\".###It is a replica of the grotto at Lourdes, "
                    "France where the Virgin Mary reputedly appeared to Saint "
                    "Bernadette Soubirous in 1858.###At the end of the main drive "
                    "(and in a direct line that connects through 3 statues and the "
                    "Gold Dome), is a simple, modern stone statue of Mary.")
        index0 = "5"
        expected_line0 = self.question0 + "\t" + context0 + "\t" + index0

        context1 = ("Immediately behind the basilica is the Grotto, a Marian "
                    "place of prayer and reflection.###It is a replica of the grotto "
                    "at Lourdes, France where the Virgin Mary reputedly appeared to "
                    "Saint Bernadette Soubirous in 1858.###Next to the Main Building "
                    "is the Basilica of the Sacred Heart.###Atop the Main Building's "
                    "gold dome is a golden statue of the Virgin Mary.###At the end "
                    "of the main drive (and in a direct line that connects through 3 "
                    "statues and the Gold Dome), is a simple, modern stone statue of "
                    "Mary.###Architecturally, the school has a Catholic "
                    "character.###Immediately in front of the Main Building and "
                    "facing it, is a copper statue of Christ with arms upraised with "
                    "the legend \"Venite Ad Me Omnes\".")
        index1 = "6"
        expected_line1 = self.question1 + "\t" + context1 + "\t" + index1

        reader = SquadSentenceSelectionReader()
        output_filepath = reader.read_file(join(self.TEST_DIR, "squad_data.json"))
        with open(output_filepath, "r") as generated_file:
            lines = []
            for line in generated_file:
                lines.append(line.strip())
        assert expected_line0 == lines[0]
        assert expected_line1 == lines[1]
