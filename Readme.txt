Environment setup:

1. Install ale-py package using the following command:
pip install ale-py

2. Install matplotlib to see the performance of the model:
pip install matplotlib

3. If you planned to run it on the troitier server, please do the following steps in extra:
	a. Navigate to the root folder of this project
	b. python3 -m venv venv
	c. source venv/bin/activate
	d. Install packages indicated in 1. and 2. using "pip install"
	REMARK: Do not turn set display to True on Troitier Server!!!

To run the program:

python3 main.py

By default it will test the trained model 1 time, to test other functionalities or to train the model please see below:

Game Board dispaly:
By default, the game Board would not be displayed, if you want to enable it to accelerate the training or testing process, go to line 564, set ale.setBool("display_screen", False) to True.

Random seed:
By default, the game would run on a random seed, if you have a specific test to train or to test, uncomment line 559 and change the number to the seed number that you want to test.

Navigate to the main function:
1. If you want to see the performance of the trained model, simply run test(number_of_test), a graph will be plotted if the number_of_test >= 10; by default it will test 1 time.

2. If you want to train a new model from scratch, please delete everythin in weights.csv, and base on your need, run greedy_train(number_of_train) or decay_train(number_of_train), by default they will train for 250 episodes; a graph would be plot at the end of training

3. If you want to train based on the existing model, simply run greedy_train(number_of_train) or decay_train(number_of_train), a graph would be plot at the end of training
