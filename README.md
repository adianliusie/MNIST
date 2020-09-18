# MNIST
This is my quick implementation of the classic MNIST problem done through [kaggle](https://www.kaggle.com/c/digit-recognizer)

## Set Up 
In ```sort.py``` I defined a basic class which helps process the csv MNIST data. This class reads the csv file and separates the rows into labels and data points which are saved in list of lists, and can do the same for the test data (except clearly without saving labels as they're unnavailable). There is also a basic method to visualise the row of the data and see each 28x28 pixel MNIST picture.

```run.py``` is the main training, evaluation and prediction script. It splits the data into a 80:20 split for the training:dev set and then uses basic SGD with momentum as the optimizer. There are a few debugging measurements made in each epoch such as training loss, epoch training accuracy, dev loss and dev accuracy to help see how the model is performing. 

```models.py``` as standard is where all the pytorch models are defined. There were 3 different models looked at, a basic neural network, a 2 layer deep neural netowrk and a convolutional netural network. 

### Basic Neural Network
The first model I built was a basic single layered Neural Network (NN). This simple model proved to solve the problem reasonably well and could achieve **90% accuracy** with a short amount of training. Since the models looks at direct correlations between each pixels and the output, this model was resistant to overfitting and the dev set accuracy performed similarly to the training set accuracy, so there was little generalisation issues but clearly this model was too simple.

### Deep Neural Network
To make the model a bit more powerful, I started with a basic 2 layer deep nerual network with a Relu activation function. As expected this resulted in model which had higher capacity and which could therefore model the problem better. This DNN managed an accuracy of **93.5%** on the test set however the generalisation error was quite significant in this task, and longer training would get the model to overfit to the training set and although it could achieve a 98% training accuracy the dev set accuracy would be rooted at 93%. 

### Convolutional Neural Network
Both previous models ignore spacial correlation and information and does not take spacial information into account. The CNN model has 2 convolutional neural networks seperated by a maxpooling layer and a final linear layer before the softmax classification. This model manages a **97.2%** accuracy which is a substancial improvement to the previous neural network. Further finetuning has shown potential to further push this accuracy, and also methods such as data augmentation with small translations or rotations could help improve accuracy. However this was all just a quick exercise for me, and although I'm aware it's common for people to ahcieve 100% or close to 100% accuracy, I just wanted to try to a brief investigation in the different models on the task and get used to kaggle so I was content to stick with the 97%. 
