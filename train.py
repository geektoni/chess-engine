from model import ChessEngine
from dataset_loader import ChessDataset

import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torch.optim import Adam

import numpy as np

from tqdm import tqdm

from argparse import ArgumentParser
import os
import datetime

if __name__ == "__main__":

    # Set up the argument parser
    parser = ArgumentParser()
    parser.add_argument("--save-model", type=str, help="Location where to save the trained model", default="./models")
    parser.add_argument("--dataset", type=str, help="Location where to search for/create the dataset",
                        default="./dataset")
    parser.add_argument("--seed", type=int, help="Set the random number generator seed to ensure reproducibility",
                        default=2020)
    parser.add_argument("--encoding", type=str, help="Select the encoding used for the chess positions (one_hot "
                                                       "or binary)", default="one_hot")
    parser.add_argument("--epochs", type=int, help="Training epochs", default=1000)
    parser.add_argument("--gpu", type=bool, help="If True, the training will procede on a GPU if it is available",
                        default=True)

    # Parse the arguments
    args = parser.parse_args()

    dataset_directory = args.dataset
    model_directory = args.save_model
    seed = args.seed
    encoding = args.encoding
    epochs = args.epochs
    gpu = args.gpu

    # Set seed and deterministic behaviour
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_deterministic(True)

    # Model name
    timestamp = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    model_save = os.path.join(model_directory,
                              "chess_engine_one_hot_{}.pth".format(timestamp))

    # Create the dataset and convert the games into something
    # more usable (one-hot encoded version)
    dataset = ChessDataset(dataset_path=dataset_directory, encoding_type=encoding)
    dataset.convert_games()

    # Split the dataset into train/test
    lengths = len(dataset)
    train_length = int(lengths*0.8)
    test_length = lengths - train_length
    train, test = torch.utils.data.random_split(dataset, [train_length, test_length])

    # Create the DataLoader object used for training
    dataloader_train = DataLoader(train, batch_size=500,
                                shuffle=True, num_workers=4,
                                pin_memory=True)

    dataloader_test = DataLoader(test, batch_size=500,
                                  shuffle=True, num_workers=4,
                                 pin_memory=True)

    # Create the model we will use
    chess_model = ChessEngine(encoding_type=encoding)

    # Define the optimizer
    criterion = MSELoss()
    optimizer = Adam(chess_model.parameters(), lr=0.001)

    # Check if we are on GPU, then move everything to GPU
    if torch.cuda.is_available() and gpu:
        device = "cuda:0"
    else:
        device = "cpu"

    # Move model to device
    chess_model.to(device)


    # Train the model
    for epoch in tqdm(range(epochs)):

        train_loss = 0.0
        for i_batch, sample_batched in enumerate(dataloader_train):

            inputs, labels = sample_batched
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = chess_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss

        # After each epoch return the validation results
        validation_loss = 0.0
        for t_batch, sample_batched_test in enumerate(dataloader_test):

            inputs, labels = sample_batched_test
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = chess_model(inputs)
            loss_validation = criterion(outputs, labels)

            validation_loss += loss_validation

        # Print the validation loss
        print("Train/Validation Loss: {:.3f} {:.3f}".format(train_loss/i_batch,
                                                            validation_loss/t_batch))

        # After each batch, we checkpoint the model
        torch.save(chess_model.state_dict(), model_save)
