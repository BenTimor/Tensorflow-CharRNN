from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from eval_utils import setup_args
from train_utils import load_configs, build_model
import tensorflow as tf

args = setup_args()
config = load_configs(args.path)

def generate_text(model, prime):
    # Evaluation step (generating text using the learned model)
    char2int = config['char2idx']
    int2char = config['idx2char']

    # Converting our start string to numbers (vectorizing)
    input_eval = [char2int[s] for s in prime]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Here batch size == 1
    model.reset_states()
    for i in range(args.len):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the word returned by the model
        predictions = predictions / args.temp
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        # We pass the predicted word as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(int2char[predicted_id])

    return prime + ''.join(text_generated)

def main_generate_text(model, prime):
    output = generate_text(model, prime)
    print(output)
    # Outputting into a file
    if args.file:
        with open(args.file, "w") as file:
            file.write(output)

def main():
    unique_chars_amount = config['vocab_size']
    embedding_dim = config['embedding']
    rnn_units = config['units']
    n_layers = config['layers']

    # Building & Loading our model
    print("Starting to build the model.")
    model = build_model(unique_chars_amount, embedding_dim, rnn_units, 1, n_layers)
    model.load_weights(tf.train.latest_checkpoint(args.path))
    model.build(tf.TensorShape([1, None]))
    print("Model is ready.")

    # If we're looping, ask for prime and generate text
    while args.loop:
        prime = input("\nEnter prime (Or empty enter to exit): ").replace("\\n", "\n")
        if not prime:
            break
        print("Starting to generate text.\n")
        main_generate_text(model, prime)
    # If we're not looping, just generate the text
    else:
        print("Generating text.\n")
        # Generating
        main_generate_text(model, args.prime)

if __name__ == '__main__':
    main()