# Arithmometer

---

## Toy Neural Network built with PyTorch and allowing to compute sum of two numbers ##

This weights and architecture of this NN will be used in my Pipeline project as an example of a "general neural network".

## Requirements

Python 3.9+ (with PyTorch + pandas)

## How to use

Generate the dataset file (totals.csv) with:

```shell
$ python ./gen_numbers.py
```

Run the trainer:

```shell
$ python ./trainer.py
```

After training finishes the weights will be saved to output file.
The file name format is:

`{number of features}ftrs_{number of epochs}epochs_{batch size}batch.pth`

Number of input features is two in our case (we compute total of two numbers).
Number of epochs and batch size can be changed in the `Trainer's` constructor.