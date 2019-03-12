import numpy as np
import os.path as path
import datetime
from keras.layers import Dense, LSTM, Input
from keras.models import Model
from keras.callbacks import TensorBoard
from utils import print_sequence
import dataset as ds
from argparse import ArgumentParser


def run(args):
    # Read the data from the corpus directory.
    encoder_input_data,\
        decoder_input_data,\
        decoder_target_data,\
        pos_to_int_dict = ds.read_data(args.corpus_dir)
    # Determine the size of POS embeddings.
    # Input data has the shape
    # (num_phrases, max_phrase_length, embedding_size).
    embedding_dim = encoder_input_data.shape[2]
    # Start creating the model
    # We define the encoder input saying that it will accept matrices of
    # unknown number of rows and `embedding_dim` columns.
    encoder_inputs = Input(shape=(None, embedding_dim))
    # The encoder is an LSTM layer
    # with the size of output space = `args.hidden_size`.
    encoder = LSTM(args.hidden_size, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # The LSTM layer will output 2 state vectors which will
    # be used to 'seed' the decoder.
    encoder_states = [state_h, state_c]
    # The decoder will also accept as input matrices of
    # unknown number of rows and `embedding_dim` columns.
    decoder_inputs = Input(shape=(None, embedding_dim))
    decoder_lstm = LSTM(
        args.hidden_size, return_sequences=True, return_state=True)
    # As mentioned before, we pass the `encoder_states` as the
    # initial 'seed' of the decoder.
    decoder_outputs, _, _ = decoder_lstm(
        decoder_inputs, initial_state=encoder_states)
    # The output from decoder needs to be passed
    # through an activation function that will determine
    # what kind of token is the output token.
    decoder_dense = Dense(embedding_dim, activation=args.activation)
    decoder_outputs = decoder_dense(decoder_outputs)
    # Setup TensorBoard visualization of model evolution.
    # Define the log directory for current run.
    current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H%M')
    run_config = "iter-{}-bs-{}-hs-{}-act-{}-{}".format(
        args.num_iterations, args.batch_size, args.hidden_size,
        args.activation, current_time)
    logdir = path.join('./logs', run_config)
    # Create the visualization callback.
    tensorboardDisplay = TensorBoard(
        log_dir=logdir,
        histogram_freq=0,
        write_graph=True,
        write_images=True,
        write_grads=True,
        batch_size=16)
    # Build and train the model.
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.fit([encoder_input_data, decoder_input_data],
              decoder_target_data,
              batch_size=args.batch_size,
              epochs=args.num_iterations,
              callbacks=[tensorboardDisplay])
    model.save("{}.h5".format(run_config))
    # Start sampling from the trained model.
    # Create an encoder that will receive the input sequence and encode it.
    encoder_model = Model(encoder_inputs, encoder_states)
    # Create a decoder model that will receive the encoded input
    # and `start-of-sequence` token.
    decoder_state_input_h = Input(shape=(args.hidden_size, ))
    decoder_state_input_c = Input(shape=(args.hidden_size, ))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs,
                          [decoder_outputs] + decoder_states)
    # Feed a random sample to the encoder to see it in action.
    # Get the sample index.
    idx = np.random.randint(encoder_input_data.shape[0])
    # Reshape the sample.
    input_seq = encoder_input_data[idx]
    input_seq = np.reshape(input_seq,
                           (1, input_seq.shape[0], input_seq.shape[1]))

    print_sequence(input_seq, pos_to_int_dict)
    print("-" * 10)
    # Get the sample encoding.
    states_value = encoder_model.predict(input_seq)
    # Create the `start-of-sequence` token to be fed to decoder.
    target_seq = np.zeros((1, 1, embedding_dim))
    idx = pos_to_int_dict['<start>']
    target_seq[0, 0, idx] = 1
    # Create a container for output sequence.
    output_seq = np.copy(target_seq)
    # Feed the decoder the encoded sequence and sample new token at each
    # step. At the next step feed the sampled token also.
    stop = False
    max_seq_len = encoder_input_data.shape[1]
    while not stop:
        output_tokens, h, c = decoder_model.predict([target_seq] +
                                                    states_value)
        token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = np.zeros((1, 1, embedding_dim))
        sampled_token[0, 0, token_index] = 1
        if token_index == pos_to_int_dict[
                '<end>'] or output_seq.shape[1] >= max_seq_len:
            stop = True
        target_seq = np.zeros((1, 1, embedding_dim))
        target_seq[0, 0, token_index] = 1
        output_seq = np.concatenate((output_seq, sampled_token), axis=1)
        states_value = [h, c]

    print_sequence(output_seq, pos_to_int_dict)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument(
        '--corpus-dir',
        help='The directory with training data.',
        required=True)
    parser.add_argument(
        '--num-iterations',
        help='Number of training iterations.',
        required=False,
        type=int,
        default=100)
    parser.add_argument(
        '--hidden-size',
        help='Number of hidden units.',
        required=False,
        type=int,
        default=128)
    parser.add_argument(
        '--activation',
        help='Name of the activation function for the decoder.',
        required=False,
        default='softmax',
        choices=[
            'softmax', 'elu', 'selu', 'relu', 'tanh', 'sigmoid', 'linear'
        ])
    parser.add_argument(
        '--batch-size',
        help='The size of the training batch.',
        required=False,
        type=int,
        default=16)
    parser.add_argument(
        '--tensorboard-log-dir',
        help='Log directory for TensorBoard.',
        required=False,
        default='./logs/')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    run(args)
