from .word import Word


class WordList:
    """ Represents a list of words.
    This class is used to manage a list of words, relying on lazy loading to
    load data from a file only when needed.
    """

    def __init__(self, file_name: str):
        """ Initializes the WordList object with a given file name.

        :param file_name: The name of the file containing the words to be
        loaded.
        """
        assert isinstance(file_name, str), 'Invalid file_name (arg #1)'
        self._file_name = file_name
        self.__word_list = None
        self.__vocab_list = None

    def _ensure_data(self):
        if self.__word_list is None:
            vocab = set()
            self.__word_list = []
            with open(self._file_name) as file:
                for word_text in file.read().splitlines():
                    self.__word_list.append(Word(word_text))
                    for char in word_text:
                        vocab.add(char)
            self.__vocab_list = list(vocab)
            self.__vocab_list.sort()
            self.__vocab_list.insert(0, '.')

    @property
    def _words(self):
        self._ensure_data()
        return self.__word_list

    @property
    def _vocabulary(self):
        self._ensure_data()
        return self.__vocab_list

    def __repr__(self) -> str:
        """ Representation of the WordList object."""
        return f'WordList(count={self.count}, vocab={self.vocabulary_size})'

    def __len__(self) -> int:
        """ Length of the word list."""
        return len(self._words)

    def __getitem__(self, index: int) -> Word:
        """ Retrieves a word from the word list.

        :param index: The index of the word to be retrieved. Raw indicies must
        be non-negative.
        :returns: The word at the specified index.
        """
        if isinstance(index, slice):
            start, stop, step = index.start, index.stop, index.step
            return self._words[start:stop:step]
        else:
            assert index >= 0, 'Invalid index (arg #1)'
            return self._words[index]

    @property
    def vocabulary(self) -> int:
        """ Returns a list of unique characters in the word list."""
        return self._vocabulary

    @property
    def vocabulary_size(self) -> int:
        """ Returns the size of the vocabulary."""
        return len(self._vocabulary)
