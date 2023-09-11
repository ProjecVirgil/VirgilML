# VirgilML ** OLD READ ME NOT UPDATED **

## Introduction 📝

I have created a model based on the SVC (Support Vector Classifier) Machine Learning algorithm in which the task is to understand the context of the sentence or command so that successively the various commands or other preliminary instructions can be executed 

## Important Notes 📋

I am really young in this field, I have just approached and I have not studied (for now) this subject namely that of model and data analysis but based on what few (but enough for now) data and tests I have done I have tried various algorithms and datasets with various strategies but I am always open to advice and suggestions on the dataset and model.

## Model alternative ✅
I evaluated and studied various models and alternatives including:

- [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html): Support Vector Classifier
- [RFC](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html): Random Forest Classifier
- [LG](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html): Logistic Regression
- [Deep Learning](https://www.ibm.com/topics/deep-learning) with neural network based on 3 layer (Embedding,LSTM,Dense) 

The library used for the model are [Sk-learn](https://scikit-learn.org/stable/index.html)(SVC,RFC,LG) and [TensorFlow](https://www.tensorflow.org/?hl=it)(Deep Learning)

## Datas and Dataset 📅

I try various type of dataset and various techniques
but to write a huge of amount of sentences for what I need is very long and complex, I tried to search online but without success in case someone has available a dataset for my problem let me know....
Back to us in the various files you can find various types of dataset with very different amounts and data the first version is

- **data_old**: Where was my first attempt to write and create a dataset unsuccessfully because the data were few and the questions too generic in this dataset you will not even find the class AL (Other) which is also the most difficult class to manage, in the same folder you will also find a beginning of "strategy" in fact here I started to remove all the words defined stopwords but the amount of data was too little and I decided to start again and go further
- **data_final**: (Now you will understand why I went to this first and not data_test)
this dataset was supposed to be the end in fact as you can see it is very neat and understandable with about 1000 sentences so a relatively larger dataset than normal (by my standards) here the removal of stopwords was done by code during the creation of the model and the dataset created here was tested and used mainly for the neural network (believing I had enough data... spoiler no)
- **data_test:** In the end this was my dataset taken as a reference because then you will see in the tests it turned out to be quite efficient despite initially 
it had a much smaller amount of data and then grew to about 1800 data (reference to the main csv file i.e. Dataset_Learning/W)
Here as you can see we have 4 different files: 
2 files son for learning and 2 for testing in turn one file is not processed and tokenize and the other one is 
The test file is about 800 data and it is a benchmark that I followed a lot initially but in the end it turned out to be inaccurate in fact I opted for other means

The tool for tokenize and clear from stop word is in the directory creation_dataset

## Now see some Graphs and Tables 📈

This is a simple graph to give you an understanding of how important the amount of data is in a machine learning algorithm such as this one

Algorithm used **SVC**

![IMG](/assets/Data_Eff.png)

Obviously the results,the efficiency and the amount of data changes from algorithm to algorithm but this I think is enough guidance to understand how important the amount of data and the quality

### Let us now turn instead to why I chose SVC as my algorithm.

This table representation of past tests for each algortim with different datasets and different date strategies

| Algorithm  | Tokenize/StopsWord | Dataset  | Testpassed on pytest |
|------------|--------------------|----------|----------------------|
| neural network | No      | DF (Data_Final) | 11/31               |
| neural network | Yes     | DF (Data_Final) | 8/31                |
| neural network | No      | DT (Data_Test ) | 23/31               |
| neural network | Yes     | DT (Data_Test ) | 13/31               |
| SVC            | No      | DF              | 25/31               |
| SVC            | No      | DT              | 27/31               |
| SVC            | Yes     | DT              | 29/31               |
| LG             | No      | DT              | 23/31               |
| LG             | Yes     | DT              | 28/31               |
| LG             | No      | DF              | 22/31               |
| RFC            | No      | DT              | 26/31               |
| RFC            | Yes     | DT              | 26/31               |
| RFC            | No      | DF              | 22/31               |

As you can see SVC with test dataset and with clean data is the one that basically had the best score and that is mainly what led me to choose this algorithm because despite the small amount of data it manages to cope discretely 
obviously the neural network I could not test it the best having little data but nothing denies in the future to make an improvement 

### Let's analyze the chosen model

![IMG](/assets/confused_matrix.png)
![IMG](/assets/test_train.png)
![IMG](/assets/CrossValidationSet.png)

As you can see from this data the model is not so efficient in some classes especially but in reatlity for now it is more than good obviously I will go to improve the dataset and make it more efficient and maybe I will also make some tweaks to some parameters of the model 

## Conclusion 🔚

In conclusion as a first approach to this world it could have been worse I managed to build a base to continue and improve the development of the virgil project, I will probably create other models surely better and I will also use them for other tasks always to improve the use and implementation of virgil, Besides I am already experimenting and trying the first tests with the model on VirgilAI
