<H3>Name : MADHUMITHA R R</H3>
<H3>Register no : 212224240083</H3>
<H3>Date : 14.02.2026</H3>
<H3>Experiment No. 2 </H3>

## Implementation of Perceptron for Binary Classification
# AIM:
To implement a perceptron for classification using Python<BR>

# EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

# RELATED THEORETICAL CONCEPT:
A Perceptron is a basic learning algorithm invented in 1959 by Frank Rosenblatt. It is meant to mimic the working logic of a biological neuron. The human brain is basically a collection of many interconnected neurons. Each one receives a set of inputs, applies some sort of computation on them and propagates the result to other neurons.<BR>
A Perceptron is an algorithm used for supervised learning of binary classifiers.Given a sample, the neuron classifies it by assigning a weight to its features. To accomplish this a Perceptron undergoes two phases: training and testing. During training phase weights are initialized to an arbitrary value. Perceptron is then asked to evaluate a sample and compare its decision with the actual class of the sample.If the algorithm chose the wrong class weights are adjusted to better match that particular sample. This process is repeated over and over to finely optimize the biases. After that, the algorithm is ready to be tested against a new set of completely unknown samples to evaluate if the trained model is general enough to cope with real-world samples.<BR>
The important Key points to be focused to implement a perceptron:
Models have to be trained with a high number of already classified samples. It is difficult to know a priori this number: a few dozen may be enough in very simple cases while in others thousands or more are needed.
Data is almost never perfect: a preprocessing phase has to take care of missing features, uncorrelated data and, as we are going to see soon, scaling.<BR>
Perceptron requires linearly separable samples to achieve convergence.
The math of Perceptron. <BR>
If we represent samples as vectors of size n, where ‘n’ is the number of its features, a Perceptron can be modeled through the composition of two functions. The first one f(x) maps the input features  ‘x’  vector to a scalar value, shifted by a bias ‘b’
f(x)=w.x+b
 <BR>
A threshold function, usually Heaviside or sign functions, maps the scalar value to a binary output:

 


<img width="283" alt="image" src="https://github.com/Lavanyajoyce/Ex-2--NN/assets/112920679/c6d2bd42-3ec1-42c1-8662-899fa450f483">


Indeed if the neuron output is exactly zero it cannot be assumed that the sample belongs to the first sample since it lies on the boundary between the two classes. Nonetheless for the sake of simplicity,ignore this situation.<BR>


# ALGORITHM:
STEP 1: Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Plot the data to verify the linear separable dataset and consider only two classes<BR>
STEP 4:Convert the data set to scale the data to uniform range by using Feature scaling<BR>
STEP 4:Split the dataset for training and testing<BR>
STEP 5:Define the input vector ‘X’ from the training dataset<BR>
STEP 6:Define the desired output vector ‘Y’ scaled to +1 or -1 for two classes C1 and C2<BR>
STEP 7:Assign Initial Weight vector ‘W’ as 0 as the dimension of ‘X’
STEP 8:Assign the learning rate<BR>
STEP 9:For ‘N ‘ iterations ,do the following:<BR>
        v(i) = w(i)*x(i)<BR>
         
        W (i+i)= W(i) + learning_rate*(y(i)-t(i))*x(i)<BR>
STEP 10:Plot the error for each iteration <BR>
STEP 11:Print the accuracy<BR>
# PROGRAM:
```python
#mport libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Define Perceptron class
class Perceptron:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self._b = 0.0  # bias
        self._w = None # weights
        self.errors_per_epoch = []

    def fit(self, X, y, n_iter=10):
        self._w = np.zeros(X.shape[1])
        self._b = 0.0
        self.errors_per_epoch = []

        for _ in range(n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.learning_rate * (target - self.predict(xi))
                self._w += update * xi
                self._b += update
                errors += int(update != 0.0)
            self.errors_per_epoch.append(errors)

    def f(self, X):
        return np.dot(X, self._w) + self._b

    def predict(self, X):
        return np.where(self.f(X) >= 0, 1, -1)

#Load dataset
df = pd.read_csv("/content/Iris.csv")  # update path if needed
print("First 5 rows:\n", df.head())
print()

#Use only 2 classes (Setosa & Versicolor) and 2 features for visualization
df = df[df['Species'].isin(['Iris-setosa', 'Iris-versicolor'])]
X = df.iloc[:, [0, 2]].values  # SepalLengthCm, PetalLengthCm
y = df['Species'].values
y = np.where(y == 'Iris-setosa', 1, -1)

#Plot data
plt.scatter(X[:50,0], X[:50,1], color='orange', marker='o', label='Setosa')
plt.scatter(X[50:100,0], X[50:100,1], color='blue', marker='x', label='Versicolor')
plt.xlabel("Sepal length")
plt.ylabel("Petal length")
plt.legend(loc='upper left')
plt.show()
print()

#Feature scaling
X[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

#Train Perceptron
classifier = Perceptron(learning_rate=0.01)
classifier.fit(X_train, y_train)

#Evaluate
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy*100, "%")

#Plot errors per epoch
plt.plot(range(1, len(classifier.errors_per_epoch)+1), classifier.errors_per_epoch, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Number of misclassifications')
plt.title('Perceptron Training Errors')
plt.show()

```

# OUTPUT:

<img width="762" height="182" alt="Screenshot 2025-09-17 181914" src="https://github.com/user-attachments/assets/0ed79a93-8dac-45b6-8b7a-b28ce5e2889f" />
<br><br>
<img width="700" height="500" alt="Screenshot 2025-09-17 181928" src="https://github.com/user-attachments/assets/d66b7bdf-14ed-40ad-a2a2-b03dc7e373d8" />
<br><br>

<img width="700" height="500" alt="Screenshot 2025-09-17 181937" src="https://github.com/user-attachments/assets/bba281c8-cbf2-4273-8446-19036008a828" />


# RESULT:
 Thus, a single layer perceptron model is implemented using python to classify Iris data set.

 
