import numpy as np
import constants


def generate_random_data(num_samples=100,
                         max_seq_length=50,
                         max_marks_per_sequence=10):
    x = np.zeros((num_samples, max_seq_length, constants.EMBEDDING_DIM))
    y = np.zeros((num_samples, max_seq_length + max_marks_per_sequence,
                  constants.EMBEDDING_DIM))
    z = np.zeros_like(y)

    for i in range(num_samples):
        # Mark start of sequences
        x[i, 0] = np.copy(constants.SEQ_START)
        y[i, 0] = np.copy(constants.SEQ_START)
        # Generate sequence length and number of marks
        seq_len = np.random.random_integers(max_seq_length)
        num_marks = np.random.randint(max_marks_per_sequence)
        # Mark end of sequences
        x[i, seq_len - 1] = np.copy(constants.SEQ_END)
        y[i, seq_len + num_marks - 1] = np.copy(constants.SEQ_END)
        z[i, seq_len + num_marks - 2] = np.copy(constants.SEQ_END)
        # Generate target marks
        mark_indices = generate_marks(seq_len, max_marks_per_sequence)
        # Fill arrays with random data
        j, k = 1, 1
        while j < seq_len - 1:
            token = np.random.randint(low=3, high=constants.EMBEDDING_DIM)
            x[i, j, token] = 1
            y[i, k, token] = 1
            z[i, k - 1, token] = 1
            if j in mark_indices:
                k = k + 1
                y[i, k] = np.copy(constants.MARK)
                z[i, k - 1] = np.copy(constants.MARK)
            k = k + 1
            j = j + 1
    return x, y, z


def generate_marks(seq_length, max_marks_per_sequence):
    num_marks = np.random.randint(max_marks_per_sequence)
    min_pos = int(seq_length * 0.1)
    max_pos = int(seq_length * 0.8)
    marks = np.random.random_integers(
        low=min_pos, high=max_pos, size=(num_marks, ))
    return marks


def print_item(item):
    if np.array_equal(item, constants.SEQ_START):
        return '<START>'
    if np.array_equal(item, constants.SEQ_END):
        return '<END>'
    if np.array_equal(item, constants.MARK):
        return '<MARK>'
    return '<tok>'


def print_sequence(sequence):
    items = np.apply_along_axis(print_item, 1, sequence)
    items = np.ravel(items)
    print(''.join(items))
