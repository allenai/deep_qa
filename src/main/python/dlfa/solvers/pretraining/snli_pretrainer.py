from .pretrainer import Pretrainer

class SnliPretrainer(Pretrainer):
    """
    An SNLI pretrainer is a Pretrainer that uses SNLI data.  This is still an abstract class; the
    only thing we do is add a load_data() method for easily getting SNLI inputs.
    """
    def __init__(self, solver, snli_file):
        super(SnliPretrainer, self).__init__(solver)
        self.snli_file = snli_file

    def load_data(self):
        # TODO(matt)
        pass
