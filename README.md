# Titanic-ML

Predict if a passenger survived the sinking of the Titanic or not. (0 or 1 value for the Survived variable.)
For each PassengerId in the test set.

2 versions using:

- SciKitLearn, running with RandomForestClassifier:

		Requires:
			- Numpy
			- Pandas
			- Sklearn

	Usage: python3 main.py [-n]

		Use '-n' to use a model already build.


- TensorFlow, running with neural network:

		Requires:
			- Numpy
			- Pandas
			- tensoflow

	Usage: python3 main.py [-v][-n]

		Use '-v' to use visualize cost.
		Use '-n' to train a new model.
