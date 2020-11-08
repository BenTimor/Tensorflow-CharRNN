import argparse
from train_utils import str2bool

def setup_args():
    parser = argparse.ArgumentParser(description='List of avaible commands.')
    # The path which you put after the --save on the train
    parser.add_argument('--path', dest="path", type=str, nargs='?', help='Path to checkpoint storage')
    # The first text in the prediction
    parser.add_argument('--prime', dest="prime", type=str, nargs='?', help='The text which comes at the start of the prediction.')
    # How much to generate
    parser.add_argument('--len', dest="len", metavar="Number (1000)", type=int, nargs='?',
                        help='Size of the generated text', default=1000)
    # File output
    parser.add_argument('--file', dest="file", metavar="File", type=str, nargs='?',
                        help='File to output into', default="")
    # Loop
    parser.add_argument("--loop", dest="loop", type=str2bool, nargs="?", help="Looping and asking for a prime instead of running once", default=False)
    # God knows
    parser.add_argument('--temp', dest="temp", metavar="Number (1.0)", type=float, nargs='?',
                        help='Low temperatures results in more predictable text.\n Higher temperatures results in more surprising text.',
                        default=1.0)

    args = parser.parse_args()

    if not ((args.prime or args.loop) and args.path):
        raise Exception("You must use --prime/--loop and --path")

    args.prime = args.prime.replace("\\n", "\n") if args.prime else None

    return args
