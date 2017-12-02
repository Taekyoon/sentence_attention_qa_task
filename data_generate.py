import argparse

from allen_squad import SquadReader
from loader import Vocabulary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--source_path', default='../data/squad/')
    parser.add_argument('--save_path', default='./data/squad/')
    parser.add_argument('--lower', default=True)
    args = parser.parse_args()

    files = ["train-v1.1.json", "dev-v1.1.json"]

    print("Read Dataset....")
    sr = SquadReader()
    train_dataset = sr.read(args.source_path + files[0], lower=args.lower)
    dev_dataset = sr.read(args.source_path + files[1], lower=args.lower)

    print("Setting Vocabulary....")
    vocab = Vocabulary()

    vocab.inject_dataset(train_dataset)
    print(vocab.word_indexer)
    vocab.inject_dataset(dev_dataset)
    print(vocab.word_indexer)

    print("Saving Dataset file...")
    vocab.save_vocab(args.save_path + 'lower_vocabulary.json')
    dev_dataset.save_dataset(file_path=args.save_path + "lower_dev_dataset.json")
    train_dataset.save_dataset(file_path=args.save_path + "lower_train_dataset.json")

if __name__ == '__main__':
    main()