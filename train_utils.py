import argparse
import os
import pickle
import tensorflow as tf
from pathlib import Path

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

def load_configs(directory):
    path = os.path.join(directory, "parameters.bin")
    return pickle.loads(open(path,'rb').read())

def save_model_configs(directory, params):
    Path(directory).mkdir(parents=True, exist_ok=True)
    path = os.path.join(directory, "parameters.bin")
    dumped = pickle.dumps(params)
    f = open(path, 'wb+')
    f.write(dumped)

def build_model(vocab_size, embedding_dim, rnn_units, batch_size, layers_amount):
    layers = [tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None])] # First layer

    for n in range(layers_amount):
        layers.append(tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform')) # Creating the rest of the layers

    layers.append(tf.keras.layers.Dense(vocab_size)) # Creating last layer

    model = tf.keras.Sequential(layers) # Converting the layers into a model
    return model

def setup_args():
    parser = argparse.ArgumentParser(description='List of avaible commands.')
    # The path where the file is
    parser.add_argument('--data', dest="data", type=str, nargs='?', help='Path to the file to train on')
    # Where it's going to save the checkpoints
    parser.add_argument('--save', dest="save", type=str, nargs='?',
                        help='Path to where the checkpoints should be saved')
    # Epochs amount
    parser.add_argument('--epochs', dest="epochs", metavar="100", type=int, nargs='?', help='Number of epochs',
                        default=100)
    # Batch size
    parser.add_argument('--batch', dest="batch", metavar="64", type=int, nargs='?', help='Batch size', default=64)
    # LSTM unit's number
    parser.add_argument('--units', dest="units", metavar="512", type=int, nargs='?', help='Number of LSTM Units',
                        default=512)
    # LSTM unit's layers
    parser.add_argument('--layers', dest="layers", metavar="3", type=int, nargs='?', help='Number of LSTM Layers',
                        default=3)
    # The maximum length of chars
    parser.add_argument('--length', dest="length", metavar="100", type=int, nargs='?',
                        help='The maximum length sentence for a single input in characters', default=100)
    # Embedding size
    parser.add_argument('--embedding', dest="embedding", metavar="128", type=int, nargs='?',
                        help='The embedding dimension size', default=128)
    # Continue from last checkpoint
    parser.add_argument("--continue", dest="cont", metavar="False", type=str2bool, nargs='?', const=True,
                        default=False, help="Continue from last save.")
    # Just for shuffling so it won't shuffle all of the text once
    parser.add_argument("--buffer", dest="buffer", metavar="10000", type=int, nargs='?',
                        default=10000, help="Buffer size to shuffle the dataset")
    # How many batches the train has to wait before notifying on process
    parser.add_argument("--notify", dest="notify", metavar="100", type=int, nargs='?',
                        default=100, help="Notify process once every X batches")
    # How much epochs it should wait before saving
    parser.add_argument("--saving_after", dest="saving_after", metavar="1", type=int, nargs='?',
                        default=1, help="How much epochs it should wait before saving")

    return parser.parse_args()