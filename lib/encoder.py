class Encoder:

    def __init__(self, vocabulary):
        assert isinstance(vocabulary,
                          (str, list)), 'Invalid vocabulary (arg #1)'
        self._char_lookup = list(vocabulary)
        self._char_map = {}
        for index, char in enumerate(self._char_lookup):
            self._char_map[char] = index

    def __repr__(self):
        return f'Encoder({len(self._char_lookup)})'

    def get_index(self, char):
        assert isinstance(char, str), 'Invalid character (arg #1)'
        return self._char_map.get(char, None)

    def get_char(self, index):
        assert index >= 0, 'Invalid index (arg #1)'
        return self._char_lookup[index]
