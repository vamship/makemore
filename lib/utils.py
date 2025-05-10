def generate_words(model, encoder, count, generator=None):
    words = []
    for _ in range(count):
        index = encoder.get_index('.')
        chars = []
        while True:
            index = model(index, generator)
            if index == 0:
                break
            chars.append(encoder.get_char(index))
        words.append(''.join(chars))
    return words

def calculate_loss(model, words):
    log_sum = 0.0
    count = 0
    for word in words:
        for pair in word.get_pairs():
            log_sum += model.get_log_likelihood(pair)
            count +=1
    return log_sum / count

def show_stats(model, words):
    for word in words:
        for pair in word.get_pairs():
            count = model.get_count(pair)
            prob = model.get_probability(pair)
            likelihood = model.get_log_likelihood(pair)
            print(f'{pair}: {count=:>8}, {prob=:.4f}, {likelihood=:.4f}')
