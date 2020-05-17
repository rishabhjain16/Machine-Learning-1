# Machine-Learning-1

Introduction to Machine learning using sklearn library. I am using KNN and Logistic regression algorithm with 10-fold cross validation
There are lots of open source machine learning packages available online for performing all kind of things. The popular ones include:

•	Tensor Flow, an open source ML framework created by Google. It incorporates with other technologies like Python, Java, C++, JavaScript and few others as well. 

•	Keras, an open source ML software designed to work on Deep learning and to simplify deep learning models. 

•	Scikit-learn, it is one of the most famous Machine Learning library. It is built in Python and can be used with other technologies as well. It includes lots of pre-build machine learning models for performing various tasks like regression, classification, data mining, clustering etc.

•	Theano, it is one of the oldest ML library and is being used by industry for deep learning.

•	Caffe- Convolutional Architecture for Fast Feature Embedding is a ML framework written in C++.

•	Torch- It is an open source library written in C which provides optimized results without causing unnecessary complexities. 

•	Accord.NET – Open source framework for ML written in C.

These are some of the popular ML Framework. For my Machine Learning Assignment, I am choosing Scikit-learn library. This is because I am already familiar with Scikit-learn library. During summer this year, I took a ‘Machine Learning A-Z’ online course on UDEMY which taught me how to use Scikit-learn for performing machine learning. Apart from that it can be used with Python. Python is one of the best open source programming tools available. Because I am already familiar with Scikit-Learn and python, I choose to do my assignment using them. Scikit-learn comes with various inbuilt Machine learning models. All you need to do is import the library and build the model with a single line of code. That’s all it takes to start studying ML with Scikit-learn and Python. It’s very flexible and documented. There is a proper documentation available for Scikit-learn making it easy to understand and implement. Sample scripts are available to understand the working and all other API’s are well documented and safeguarded. It comes with various other tools like NumPy, Pandas, matplotlib which are combined with other machine learning packages to play with data, visualize content, create meaningful graphs or to debug a problem. Therefore, to sum it up, Scikit-learn is one of the easiest and productive way to getting to start with Machine Learning programming. Its very well documented with sample scripts available for easy understanding. It is a robust library and provides various range of supervised, unsupervised, semi-supervised and reinforcement learning algorithms. It is built upon SciPy and therefore used along with various other libraries. Also due to my familiarity with python and Scikit-learn from before, I choose to do my machine learning assignment with Scikit-Learn library.  

## Main features of Scikit-learn library include:

1. Contains popular algorithm and libraries.

2. Contains popular packages like NumPy, Pandas, Matplotlib, SciPy which are popularly used in ML 

3. Data Mining and Data Analysis

4. Open Source – BSD Licence

5. Reusable and easy implementation

6. Well written documentation

7. Can perform all kind of ML tasks like Preprocessing, Model Selection, Classification, Regression, Clustering, Dimensionality Reduction etc.

## Preparing the data for ML package:
1.	Download the “hazelnuts.txt” file from blackboard.
2.	Open excel and import the .txt file.
3.	Use excel to Delimit the file as “Characters such as commas or tabs separate each field”.
4.	Transpose the delimited files from rows to columns and columns to rows.
5.  Add the title to each file 
6.  Save the file as .CSV file. The data preparation to input data in ML package is complete. I have saved the .CSV file as Data.csv.

## ALGORITHM 1 – K NEAREST NEIGHBOUR (KNN) WITH 10-FOLD CROSS VALIDATION
K-Nearest Neighbour is a supervised learning Machine learning model. Supervised learning models works when Data is already provided. It learns from Data given by training the model on test data. After the training is complete, the test data is inputted to predict the output values. This model works by taking a data point and looking for the K-closest neighbour to that data point. K can be any number from 1 to n. Accuracy of the model varies depending on the value of K. For my assignment, I am taking k as 3. But we can use another algorithm to find the best value of k using Scikit-learn and get the most accurate model of prediction. After that, most of the data point are given a label and clustered accordingly. For our assignment, we are using Scikit learn to predict the dataset for accuracy. After that we are performing 10-fold cross validation on our dataset to predict more accuracy based on 10 different test data to get the best validation for our dataset. 
knn.predict predict the values for test data based on knn trained model. Knn.score print the accuracy of the model which in our case is coming to be 95%. Later we are using Cross validation over our KNN model  to get the best parameters for our retrained model and to avoid the situation of overfitting. print(cv_scores) prints the cross validation result of 10-fold cross validation and later their mean is calculated to get an average value of accuracy after performing the 10-fold cross validation which is coming to be 84%. Cross Validation is performed on 10 different test data to predict 10 different accuracy based on the test data to get more validate result for our dataset.

## ALGORITHM 2 – LOGISTIC REGRESSION WITH 10-FOLD CROSS VALIDATION
Logistic Regression is a predictive linear model. It is one of the most commonly used machine learning models and used in lots of business applications. It explains the relationship between a dependent variable(y) and one or more independent variable(x). It uses a sigmoid function to predict the output which is usually between 0 and 1. It also uses given data to create and train a model and then use that model to predict the test data. For our assignment, we are using Scikit learn to predict the test dataset and accuracy of the trained logistic regression model. After that we are performing 10-fold cross validation on our dataset to predict more accuracy based on 10 different test data to get the most validated accuracy for our dataset.logisticRegr.predict predict the values for test data based on logistic regression trained model. logisticRegr.score print the accuracy of the model which in our case is coming to be 92.6%. Later we are using Cross validation over our KNN model  to get the best parameters for our retrained model and to avoid the situation of overfitting. print(cv_scores) prints the cross validation result of 10-fold cross validation and later their mean is calculated to get an average value of accuracy after performing the 10-fold cross validation which is coming to be 91%. Cross Validation is performed on 10 different test data to predict 10 different accuracy based on the test data to get more validate result for our dataset.

## CONCLUSION:
In this assignment, I am using two machine learning algorithms on “hazelnuts.txt”. First, Data is being prepared in excel to be imported in our machine learning model. Then using Spyder tool, we are using Python language to add our dataset. Then we are using Pandas and NumPy to read the csv file. After that we are using sklearn to split the data into training and test data, making the machine learning model and predicting the output along with the accuracy of the model. In the end, 10-fold Cross Validation is performed on the model to predict 10 more accuracies of the model on 10 different test data. And then their mean is taken to get a more precise result. First algorithm used is K-Nearest neighbour and the second algorithm used is Logistic Regression. Now we will be comparing the results based on both algorithms.

From the result, we can see that KNN given the accuracy of 95% and Logistic Regression predicts an accuracy of 92%. Based on this prediction we can say that, KNN might be a better algorithm than Logistic Regression for prediction the variety of out hazelnuts dataset. As the dataset contains only 201 values, the model is trained on 160 values and testing is done on 41 values. It won’t be a good criterion to access the model based on this single test. Therefore, later we performed a 10-fold Cross Validation on both our algorithm to get a different set of accuracies for different set of test data build by Cross Validation as cv=10. Taking mean of these accuracy for comparing both the algorithms, we found that KNN gave an average accuracy of 84% after 10-fold cross validation while Logistic Regression gave an average accuracy of 91%. After performing 10-fold Cross Validation, we got a better set of results based on 10 test data. Hence, we can now conclude that accuracy of Logistic Regression (91%) algorithm is better than the accuracy of K-Nearest Neighbour algorithm (84%). Therefore, Logistic Regression is a better classification algorithm than KNN for our ‘hazelnuts.txt’ dataset.


## REFERENCES AND ACKOWLEDGEMENT: 

•	https://www.udemy.com/course/machinelearning/ (Machine Learning A-Z™: Hands-On Python & R in Data Science)

•	https://machinelearningmastery.com/a-gentle-introduction-to-scikit-learn-a-python-machine-learning-library/

•	https://www.thelearningmachine.ai/tree-id3

•	https://medium.com/@Mandysidana/machine-learning-types-of-classification-9497bd4f2e14

•	https://scikit-learn.org/stable/modules/cross_validation.html 

•	https://scikit-learn.org/stable/modules/preprocessing.html

•	https://machinelearningmastery.com/prepare-data-machine-learning-python-scikit-learn/

•	https://towardsdatascience.com/building-a-k-nearest-neighbors-k-nn-model-with-scikit-learn-51209555453a

•	http://www.insightsbot.com/blog/1uOwGy/python-logistic-regression-with-scikit-learn

•	https://scikit-learn.org/stable/index.html
