type PairList = list[tuple[tuple[...], tuple[...]]]

class Word:
    """ Represents a word; introduces termination characters at the beginning
    and end of the word. """

    def __init__(self, text: str):
        """ Initializes the Word object with a given text. Internal
        representations automatically add start and termination characters to
        the input text.

        :param text: The text to be represented by the Word object.
        """
        assert isinstance(text, str), 'Invalid text (arg #1)'

        self._text = '.' + text + '.'
        self._pairs = None
        self._input_count = -1

    @property
    def text(self) -> str:
        """ Text  of the word excluding the termination characters. """
        return self._text[1:-1]

    def __repr__(self) -> str:
        """ String representation of the Word object. """
        return f'Word({self._text})'

    def __str__(self) -> str:
        """ Text of the object including termination characters. """
        return self._text

    def __len__(self) -> int:
        """ Length of the word, including the termination characters. """
        return len(self._text)

    def __getitem__(self, index: int | slice) -> str:
        """ Retrieves a slices of the text represented by the word.

        :param index: The index or slice to be retrieved from the text. Raw
        indicies must be non-negative.
        """
        if isinstance(index, slice):
            start, stop, step = index.start, index.stop, index.step
            return self._text[start:stop:step]
        else:
            assert index >= 0, 'Invalid index (arg #1)'
            return self._text[index]

    def get_pairs(self, input_count: int = 1) -> PairList:
        """ Splits the words into pairs comprising inputs and labels.

        Inputs comprise a contiguous sequence of characters determined by the
        input_count arg, and labels are the next character in the sequence.

        :param input_count: The number of characters to be used as input.

        :returns:  A list of pairs represented as tuples. Each pair in turn
        contains two tuples - the first being the inputs and the second being
        the label.
        """
        if self._pairs is None or self._input_count != input_count:
            self._input_count = input_count
            self._pairs = []
            text = list(self._text[:])
            for index in range(len(text) - input_count):
                inputs = tuple(text[index:index + input_count])
                label = tuple(text[index + input_count])
                self._pairs.append((inputs, label))

        return self._pairs
