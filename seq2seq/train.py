import keras as ks
import tensorflow as tf
import numpy as np

batch_size = 64
epochs = 1
num_samples = 10000
latent_dim = 256
path = './fra.txt'


def generate_seq(num_samples):
    seq = []
    seq_tar_in = []
    seq_tar_tar = []
    for _ in range(num_samples):
        rd = np.random.randint(1,50)
        tmp = list(range(rd, rd+10))
        seq += tmp
        tmp1 = [-1] + tmp
        seq_tar_in += tmp1
        tmp2 = tmp + [-2]
        seq_tar_tar += tmp2

    seq = np.array(seq).reshape((-1, 10, 1))
    seq_tar_in = np.array(seq_tar_in).reshape((-1, 11, 1))
    seq_tar_tar = np.array(seq_tar_tar).reshape((-1, 11, 1))

    data = (seq, seq_tar_in, seq_tar_tar)
    shape = (1, 1)

    return data, shape


def generate_seq_nmt(data_path):
    # Vectorize the data.
    input_texts = []
    target_texts = []
    input_characters = set()
    target_characters = set()
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    for line in lines[: min(num_samples, len(lines) - 1)]:
        input_text, target_text = line.split('\t')
        # We use "tab" as the "start sequence" character
        # for the targets, and "\n" as "end sequence" character.
        target_text = '\t' + target_text + '\n'
        input_texts.append(input_text)
        target_texts.append(target_text)
        for char in input_text:
            if char not in input_characters:
                input_characters.add(char)
        for char in target_text:
            if char not in target_characters:
                target_characters.add(char)

    input_characters = sorted(list(input_characters))
    target_characters = sorted(list(target_characters))
    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    print('Number of samples:', len(input_texts))
    print('Number of unique input tokens:', num_encoder_tokens)
    print('Number of unique output tokens:', num_decoder_tokens)
    print('Max sequence length for inputs:', max_encoder_seq_length)
    print('Max sequence length for outputs:', max_decoder_seq_length)

    input_token_index = dict(
        [(char, i) for i, char in enumerate(input_characters)])
    target_token_index = dict(
        [(char, i) for i, char in enumerate(target_characters)])

    encoder_input_data = np.zeros(
        (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
        dtype='float32')
    decoder_input_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')
    decoder_target_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.
        for t, char in enumerate(target_text):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data[i, t, target_token_index[char]] = 1.
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.

    data = (encoder_input_data, decoder_input_data, decoder_target_data)
    shape = (num_encoder_tokens, num_decoder_tokens)
    return data, shape


# data, shape = generate_seq_nmt(path)
data, shape = generate_seq(num_samples)
encoder_input_data, decoder_input_data, decoder_target_data = data
num_encoder_tokens, num_decoder_tokens = shape

encoder_inputs = ks.Input(shape=(None, num_encoder_tokens))
encoder = ks.layers.LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)

encoder_states = [state_h, state_c]

decoder_inputs = ks.Input(shape=(None, num_decoder_tokens))
decoder_lstm = ks.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

decoder_dense = ks.layers.Dense(num_decoder_tokens)
decoder_outputs = decoder_dense(decoder_outputs)

model = ks.models.Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='rmsprop', loss='mean_squared_error')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size, epochs=epochs)

print(encoder_input_data.shape)
print(encoder_input_data[:1].flatten())
print(model.predict([encoder_input_data[:1], decoder_input_data[:1]]).flatten())
tin = np.array([[[10],[11],[12]]])
tin_t = np.array([[[-1],[10],[11],[12]]])
print(model.predict([tin,tin_t]).flatten())

encoder_model = ks.models.Model(encoder_inputs, encoder_states)

decoder_state_input_h = ks.Input(shape=(latent_dim,))
decoder_state_input_c = ks.Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = ks.models.Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

def decode_seq(input_seq):
    states_value = encoder_model.predict(input_seq)

    target_seq = np.array([[[-1]]])

    stop = False
    decoded_seq = []
    while not stop:
        output, h, c = decoder_model.predict(
            [target_seq] + states_value)

        decoded_seq.append(output.flatten()[0])

        if output < 0:
            stop = True

        target_seq[0,0,0] = output[0,0,0]

        states_value = [h, c]

    print(decoded_seq)

decode_seq(np.array([[[10],[11],[12],[13],[14]]]))
