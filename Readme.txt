Environment setup:

1. Install ale-py package using the following command:
pip install ale-py

2. Install matplotlib to see the performance of the model:
python -m pip install -U matplotlib

To run the program:
Simply navigate to the main function:
1. If you want to see the performance of the trained model, simply run test(number_of_test), a graph will be plotted if the number_of_test >= 10

2. if you want to train a new model from scratch, please delete everythin in weights.csv, and base on your need, run greedy_train(number_of_train) or decay_train(number_of_train)