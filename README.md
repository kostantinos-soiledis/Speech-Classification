# Speech-Classification
In this challenge the task was to learn to recognize which of several English words is pronounced in an audio recording.

# Chosen Model & Accuracy
Our selected best performing model is a combination of 4 layers of a 2D Convolutional Neural Network and 1 layer of a Recurrent Neural Network. Finally, there is a fully connected dense layer with 35 filters. The test accuracy reported is 0.933. 

# Research Procedure & Learning Algorithms Tried
- First, we ran a research about suitable models for multi-class audio classification. Our first candidates were Random Forest, Support Vector Machine, K-nearest Neighbours,  Convolutional Neural Network and Recurrent Neural Network. We decided to develop in parallel KNN, CNN and RNN models to see which performed the better.
- About KNN, it yielded a very low accuracy of 0.359.
- About CNN, we realized that the data is in the form of distinguishable images and can be well perceived by the specific algorithm. We decided to prioritize a Conv2D model, after seeing higher accuracies than Conv1D. The model’s highest test accuracy in codalab was 0.92.
- About RNN, we followed the idea mentioned in [2] with the addition of two more LSTM layers, since RNN excel in capturing patterns in sequential data. The model’s accuracy was 0.905
- Then, we thought we could mix RNN with CNN since CNN is good in pattern recognition and RNN implements the time component needed for the classification.We added an LSTM neuron in the Conv2D model. The model’s testing accuracy was 0.933. 

# Feature Engineering

We transformed the shape of the data from a nested list to a 3D array, and then to a 4D array which is the required dimension for our model.  Furthermore, we tried to standardize the observation’s coefficient, with (1) RobustScaler; and (2) StandardScaler. We observed  no substantial difference in accuracy. Finally we use Batch normalization for each layer.

# Parameter Tuning

Neural networks are known for their large number of hyperparameters that can be tuned to achieve better validation accuracy. Some of them are: number of layers, filters, learning rate and batch size. We defined the model as a function with learning rate (0.001, 0.0001) and batch size (32, 64) as inputs (Grid search). Then, as the number of layers and number of filters were changed manually, we obtained 4 different validation accuracy scores and compared the results. This process eventually led to our final model.

# Discussion of the Solution’s Performance
- Our model yielded a high predictive performance, given different trial and error approaches that we did:
We had developed Conv2D and RNN models independently, and thought of combining them to achieve a better score, as we have read in [1] and discussed above.
- Since we were dealing with time series, we thought of using LSTM neurons.
- We did batch normalization because it prevents overfitting and improves speed, performance and stability.
- We used various dropout values after each layer to prevent overfitting.
- We performed downsampling, starting from 351 filters and concluding to 39. 
- We began using a kernel size of 2 for each neuron, but then we increased the size for some neurons, to further capture the complexity of the problem.
- Through grid search we found that our batch size (32) and learning rate (0.001) captured the pattern better.
