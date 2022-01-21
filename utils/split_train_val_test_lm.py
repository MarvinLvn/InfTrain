import argparse
import os
import random
import sys

random.seed(10)


def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Given a file containing lines, split lines across 3 files containing'
                                                 'the train, dev, and test set.')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to the input file containing quantized units.')
    parser.add_argument('--val_prop', type=float, default=0.1,
                        help='Proportion of the validation set.')
    parser.add_argument('--test_prop', type=float, default=0.1,
                        help='Proportion of the test set.')
    parser.add_argument('--prefix', type=str, default='fairseq',
                        help='Prefix to add to the basename of created files.')
    return parser.parse_args(argv)


def main(argv):
    # Args parser
    args = parseArgs(argv)

    # Read data
    data = []
    for line in open(args.input_file, 'r').readlines():
        data.append(line)

    size_train = int((1-args.val_prop - args.test_prop) * len(data))
    size_val = int(args.val_prop * len(data))
    data_train, data_val, data_test = data[:size_train], data[size_train:size_train+size_val], data[size_train+size_val:]


    data = {'train': data_train,
            'val': data_val,
            'test': data_test}

    for key, data in data.items():
        output_file = os.path.join(os.path.dirname(args.input_file), "%s_%s.txt" % (args.prefix, key))
        with open(output_file, 'w') as fin:
            for line in data:
                fin.write(line)
    print("Done splitting input file into train/dev/test sets.")

if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)

