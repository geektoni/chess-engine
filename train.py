from model import ChessEngine
from dataset_loader import ChessDataset

import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torch.optim import SGD


if __name__ == "__main__":

    # Create the dataset and convert the games into something
    # more usable (one-hot encoded version)
    dataset = ChessDataset()
    dataset.convert_games()

    # Create the DataLoader object used for training
    dataloader = DataLoader(dataset, batch_size=100,
                            shuffle=True, num_workers=0)

    # Create the model we will use
    chess_model = ChessEngine()

    # Define the optimizer
    criterion = MSELoss()
    optimizer = SGD(chess_model.parameters(), lr=0.001, momentum=0.9)

    # Train the model
    for epoch in range(500):

        epoch_loss = 0.0
        for i_batch, sample_batched in enumerate(dataloader):

            inputs, labels = sample_batched

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = chess_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print the training loss
            epoch_loss += loss.item()
            if i_batch%2000 == 0:
                print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i_batch + 1, epoch_loss/2000))
                epoch_loss = 0.0

        # After each batch, we checkpoint the model
        torch.save(chess_model.state_dict(), "./models/chess_model.pth")
