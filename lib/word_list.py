from .word import Word


class WordList:

    def __init__(self, file_name):
        assert isinstance(file_name, str), 'File name must be a string'
        self._file_name = file_name
        self.__word_list = None
        self._vocabulary = None

    @property
    def _words(self):
        if self.__word_list is None:
            vocab = set()
            self.__word_list = []
            with open(self._file_name) as file:
                for word_text in file.read().splitlines():
                    self.__word_list.append(Word(word_text))
                    for char in word_text:
                        vocab.add(char)
            self._vocabulary = list(vocab)
            self._vocabulary.sort()

        return self.__word_list

    def __repr__(self):
        return f'WordList({len(self.names)})'

    def __getitem__(self, index):
        assert index >= 0, 'Word index must be greater than or equal to zero'
        return self._words[index]

    @property
    def count(self):
        return len(self._words)

    @property
    def vocabulary(self):
        return self._vocabulary
