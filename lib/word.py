class Word:

    def __init__(self, text):
        assert isinstance(text, str), 'Invalid text (arg #1)'
        self._text = '.' + text + '.'
        self._pairs = None

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

    def get_pairs(self):
        if self._pairs is None:
            self._pairs = []
            text = list(self._text[:])
            for first, second in zip(text[:], text[1:]):
                self._pairs.append((first, second))
        return self._pairs
