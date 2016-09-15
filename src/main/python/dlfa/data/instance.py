class Instance:
    """
    A data instance, used either for training a neural network or for testing one.
    """
    def __init__(self, label, index: int=None):
        """
        label: Could be boolean or an index.  For simple Instances (like TextInstance), this is
            either True, False, or None, indicating whether the instance is a positive, negative or
            unknown (i.e., test) example, respectively.  For QuestionInstances or other more
            complicated things, is a class index.
        index: if given, must be an integer.  Used for matching instances with other data, such as
            background sentences.
        """
        self.label = label
        self.index = index

    @staticmethod
    def _check_label(label: bool, default_label: bool):
        if default_label is not None and label is not None and label != default_label:
            raise RuntimeError("Default label given with file, and label in file doesn't match!")
