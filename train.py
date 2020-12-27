from model import ChessEngine, ChessEngine2, ChessEngineBinary
from dataset_loader import ChessDataset

import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torch.optim import SGD

import numpy as np

from tqdm import tqdm


if __name__ == "__main__":

    # Set seed and deterministic behaviour
    torch.manual_seed(2020)
    np.random.seed(2020)
    torch.set_deterministic(True)

    # Model name
    model_save = "./chess_engine_binary.pth"

    # Create the dataset and convert the games into something
    # more usable (one-hot encoded version)
    dataset = ChessDataset(encoding_type="binary")
    dataset.convert_games()

    # Split the dataset into train/test
    lengths = len(dataset)
    train, test = torch.utils.data.random_split(dataset, [int(lengths*0.8), int(lengths*0.2)])

    # Create the DataLoader object used for training
    dataloader_train = DataLoader(dataset, batch_size=100,
                                shuffle=True, num_workers=0)

    dataloader_test = DataLoader(dataset, batch_size=1,
                                  shuffle=True, num_workers=0)

    # Create the model we will use
    chess_model = ChessEngineBinary(encoding_type="binary")

    # Define the optimizer
    criterion = MSELoss()
    optimizer = SGD(chess_model.parameters(), lr=0.001, momentum=0.9)

    # Check if we are on GPU, then move everything to GPU
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    # Move model to device
    chess_model.to(device)


    # Train the model
    for epoch in tqdm(range(1000)):

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

            # Use
            if t_batch >= 10:
                break

            inputs, labels = sample_batched_test
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = chess_model(inputs)
            loss_validation = criterion(outputs, labels)

            validation_loss += loss_validation

        # Print the validation loss
        print("Train/Validation Loss: {:.3f} {:.3f}".format(train_loss/i_batch, validation_loss/10))

        # After each batch, we checkpoint the model
        torch.save(chess_model.state_dict(), model_save)
