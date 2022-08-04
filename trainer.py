import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from model import Model

class Trainer:
    def __init__(self, dataset_name: str, test_data_size = 0.3, seed = 5, epochs = 20,
        shuffle_train = True, shuffle_test = True, batch_size_train = 5, batch_size_test = 1):

        self.__test_data_size = test_data_size
        self.__seed = seed
        self.__epochs = epochs
        self.__shuffle_train = shuffle_train
        self.__shuffle_test = shuffle_test
        self.__batch_size_train = batch_size_train
        self.__batch_size_test = batch_size_test

        self.__features_count = None
        self.__X_train = None
        self.__y_train = None
        self.__X_test = None
        self.__y_test = None

        self.__df = pd.read_csv(dataset_name)

    def prepare_training_loaders(self):
        self.__features_count = 2
        X = self.__df[['a', 'b']].values
        y = self.__df[['total']].values

        self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(
            X, y, test_size = self.__test_data_size, random_state = self.__seed)

        self.__num_samples = y.shape[0]
        
        X_train = torch.from_numpy(self.__X_train)
        y_train = torch.from_numpy(self.__y_train).view(-1, 1)

        X_test = torch.from_numpy(self.__X_test)
        y_test = torch.from_numpy(self.__y_test).view(-1, 1)

        train = torch.utils.data.TensorDataset(X_train, y_train)
        test = torch.utils.data.TensorDataset(X_test, y_test)

        self.__train_loader = torch.utils.data.DataLoader(
            train, batch_size = self.__batch_size_train, shuffle = self.__shuffle_train)
        self.__test_loader = torch.utils.data.DataLoader(
            test, batch_size = self.__batch_size_test, shuffle = self.__shuffle_test)

    def train(self):
        # Create model
        self.__nn_model = Model(input_dim = self.__features_count, 
            output_dim = 1)
        
        self.__loss_fn = nn.MSELoss()
        optimizer = optim.SGD(self.__nn_model.parameters(), 
            lr=0.01, weight_decay= 1e-5, momentum = 0.8)

        # Prepare model for training
        self.__nn_model.train() 

        for epoch in range(self.__epochs):
            trainloss = 0.0
            total_error = 0.0
            total = 0

            for train, target in self.__train_loader:
                train_batch = Variable(train).float()
                target_batch = Variable(target).float()

                # Clear previous gradients
                optimizer.zero_grad() 

                # Forward step
            
                # Compute model output
                output_batch = self.__nn_model(train_batch)

                # Calculate loss
                loss = self.__loss_fn(output_batch, target_batch)

                # Backward step

                # Compute gradients of all variables with respect to loss
                loss.backward() 

                # nn.utils.clip_grad_value_(self.__nn_model.parameters(), clip_value=1.0)

                # Perform updates using calculated gradients
                optimizer.step()  

                # Compute results

                trainloss += loss.item()

                total_error = torch.sum(torch.abs(torch.subtract(target_batch, output_batch))).item()
                total += len(target_batch)

            # Collect current epoch data
            avg_error = total_error / float(total)

            msg = "Epoch: {}/{} \tTraining Loss: {:.5f}\t Avg Error: {:.8f}".format(
                epoch + 1, 
                self.__epochs,
                trainloss,
                avg_error)
            print(msg)

        print("---")

        with torch.set_grad_enabled(False):
            testloss = 0.0
            total_error = 0.0
            total = 0

            for test, target in self.__test_loader:
                test_batch = Variable(test).float()
                target_batch = Variable(target).float()

                output_batch = self.__nn_model(test_batch)

                loss = self.__loss_fn(output_batch, target_batch)
                testloss += loss.item()

                total_error = torch.sum(torch.abs(torch.subtract(target_batch, output_batch))).item()
                total += len(target_batch)

            # Collect current epoch data
            avg_error = total_error / float(total)

        msg = "Validation: \tLoss: {:.5f}\t Avg Error: {:.8f}".format(
            testloss,
            avg_error)
        print(msg)

        name = f"{self.__features_count}ftrs_{self.__epochs}epochs_{self.__batch_size_train}batch.pth"
        torch.save(self.__nn_model.state_dict(), name)

def main():
    trainer = Trainer('totals.csv')
    trainer.prepare_training_loaders()
    trainer.train()

if __name__ == '__main__':
    main()
