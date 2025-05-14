class Word:

    def __init__(self, text):
        assert isinstance(text, str), 'Invalid text (arg #1)'
        self._text = '.' + text + '.'
        self._pairs = None
        self._input_count = -1

    @property
    def text(self):
        return self._text

    def __repr__(self):
        return f'Word({self._text})'

    def __str__(self):
        return self._text

    def __len__(self):
        return len(self._text)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return Word(self._text[index])
        else:
            assert index >= 0, 'Invalid index (arg #1)'
            return self._text[index]

    def get_pairs(self, input_count=1):
        if self._pairs is None or self._input_count != input_count:
            self._input_count = input_count
            self._pairs = []
            text = list(self._text[:])
            for index in range(len(text) - input_count):
                inputs = tuple(text[index:index + input_count])
                label = tuple(text[index + input_count])
                self._pairs.append((inputs, label))

        return self._pairs
