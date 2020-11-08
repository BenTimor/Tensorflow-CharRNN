from os import path, environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from train_utils import *
import numpy as np
import tensorflow as tf
import time

args = setup_args() # Command line arguments

@tf.function
def train_step(model, optimizer, inp, target):
    with tf.GradientTape() as tape:
        predictions = model(inp)
        loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                target, predictions, from_logits=True))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss

def load_data_file():
    """
    Loading all the information needed from the data file
    """
    # Loading data
    chars = open(args.data, "rb").read().decode("utf-8")
    unique_chars = sorted(set(chars))
    # Creating a dict & array which allows you to convert chars into numbers and back
    # Then using this dict to convert the data file into numbers (& converting it into a dataset) and calculating examples per epoch
    chars_length = len(chars)
    unique_chars_amount = len(unique_chars)
    char2int = {u: i for i, u in enumerate(unique_chars)}
    int2char = np.array(unique_chars)
    steps_per_epoch = chars_length//args.length//args.batch

    dataset = tf.data.Dataset.from_tensor_slices( # Converting the numpy array into a tensorflow's dataset
        np.array([char2int[c] for c in chars]) # Converting the data file chars into numbers
    ).batch(
        args.length+1, drop_remainder=True # Converting to batches
    ).map(
        split_input_target
    ).shuffle(args.buffer).batch(args.batch, drop_remainder=True) # Shuffling the data

    return chars_length, unique_chars_amount, char2int, int2char, steps_per_epoch, dataset

def load_saved_config():
    """
    Loading all the information needed from the saved config (if exists and needed)
    """
    confs = None
    if path.exists(args.save) and args.cont:
        # Load last configuration if user wants to continue from last checkpoint
        try:
            confs = load_configs(args.save)
            embedding_dim = confs['embedding']
            rnn_units = confs['units']
            n_layers = confs['layers']
        except Exception as _:
            print("There was an error loading the last configuration. If you want to start over, remove the parameter '--continue TRUE'.")
            return
    else:
        # Else, use the command line arguments for configuration
        embedding_dim = args.embedding
        rnn_units = args.units
        n_layers = args.layers

    return confs, embedding_dim, rnn_units, n_layers

def main():
    print("Starting to load the model...")

    checkpoint_path = path.join(args.save, "ckpt_{epoch}") # Setuping checkpoints path
    chars_length, unique_chars_amount, char2int, int2char, steps_per_epoch, dataset = load_data_file() # Loading the data
    confs, embedding_dim, rnn_units, n_layers = load_saved_config() # Loading config

    model = build_model(unique_chars_amount, embedding_dim, rnn_units, args.batch, n_layers)

    if path.exists(args.save) and args.cont:
        try:
            model.load_weights(tf.train.latest_checkpoint(args.save)) # Load last checkpoint if needed
        except Exception as _:
            print('Error loading checkpoints')

    if confs is None:
        confs = {
            'units': args.units,
            'embedding': args.embedding,
            'layers': args.layers,
            'vocab_size': unique_chars_amount,
            'char2idx': char2int,
            'idx2char': int2char,
        }
        save_model_configs(args.save, confs) # Saving parameters

    print(f"Length of data file: {chars_length} | Unique chars: {unique_chars_amount}")
    print("Starting to train the model...")

    optimizer = tf.keras.optimizers.Adam()

    for epoch in range(args.epochs):
        start = time.time()

        # initializing the hidden state at the start of every epoch
        model.reset_states()
        print()
        for (batch, (inp, target)) in enumerate(dataset):
            loss = train_step(model, optimizer, inp, target)
            if batch % args.notify == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch}/{steps_per_epoch}] Loss {loss}")
        print()
        if (epoch + 1) % args.saving_after == 0:
            model.save_weights(checkpoint_path.format(epoch=epoch))
            print("Model Saved!")

        print(f"Finished Epoch {epoch+1}/{args.epochs}. Time taken: {time.time()-start} seconds.")

    print("Model has been fully trained!")
    model.save_weights(checkpoint_path.format(epoch=epoch))

if __name__ == '__main__':
    main()