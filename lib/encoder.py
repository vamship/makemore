class Encoder:

    def __init__(self, vocabulary):
        assert isinstance(vocabulary,
                          (str, list)), 'Vocabulary must be an array or string'
        self._char_lookup = list(vocabulary)
        self._char_map = {}
        for index, char in enumerate(self._char_lookup):
            self._char_map[char] = index

    def __repr__(self):
        return f'Encoder({len(self._char_lookup)})'

    def get_index(self, char):
        assert isinstance(char, str), 'Character input must be a string'
        return self._char_map.get(char, None)

    def get_char(self, index):
        assert index >= 0, 'Character index must be greater than or equal to 0'
        return self._char_map[index]
