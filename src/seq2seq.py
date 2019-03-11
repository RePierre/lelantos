import numpy as np
import os.path as path
import datetime
from keras.layers import Dense, LSTM, Input
from keras.models import Model
from keras.callbacks import TensorBoard
from utils import generate_random_data, print_sequence
import constants



encoder_input_data,\
    decoder_input_data,\
    decoder_target_data = generate_random_data(num_samples=100)

encoder_inputs = Input(shape=(None, constants.EMBEDDING_DIM))
encoder = LSTM(constants.LATENT_DIM, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, constants.EMBEDDING_DIM))
decoder_lstm = LSTM(
    constants.LATENT_DIM, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(
    decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(constants.EMBEDDING_DIM, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H%M')
logdir = path.join('./logs', current_time)
tensorboardDisplay = TensorBoard(
    log_dir=logdir,
    histogram_freq=0,
    write_graph=True,
    write_images=True,
    write_grads=True,
    batch_size=16)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.summary()
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data],
          decoder_target_data,
          batch_size=16,
          epochs=100,
          callbacks=[tensorboardDisplay])

encoder_model = Model(encoder_inputs, encoder_states)
decoder_state_input_h = Input(shape=(constants.LATENT_DIM, ))
decoder_state_input_c = Input(shape=(constants.LATENT_DIM, ))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs,
                      [decoder_outputs] + decoder_states)

input_seq = encoder_input_data[1]
input_seq = np.reshape(input_seq, (1, input_seq.shape[0], input_seq.shape[1]))
states_value = encoder_model.predict(input_seq)
target_seq = np.zeros((1, 1, constants.EMBEDDING_DIM))
target_seq[0, 0] = np.copy(constants.SEQ_START)
output_seq = np.copy(target_seq)

stop = False
while not stop:
    output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
    token_index = np.argmax(output_tokens[0, -1, :])
    sampled_token = np.zeros((1, 1, constants.EMBEDDING_DIM))
    sampled_token[0, 0, token_index] = 1

    if np.array_equal(constants.SEQ_END,
                      sampled_token[0, 0]) or target_seq.shape[1] > 30:
        stop = True
    target_seq = np.zeros((1, 1, constants.EMBEDDING_DIM))
    target_seq[0, 0, token_index] = 1
    output_seq = np.concatenate((output_seq, sampled_token), axis=1)
    states_value = [h, c]

print_sequence(input_seq)
print("-" * 10)
print_sequence(output_seq)
