# Active-Machine-Learning
active learning for text data annotation
From NITK, Reach me at lanceabhishek321@gmail.com for more details 
Abstract:- Advancement in the field of NLU and sentiment analysis there is a requirement of annotated dataset to develop an accurate model. With active learning we can make this process of data annotation quicker and mitigate human interaction. In Pool based active learning we collect samples and annotate it, then use it as a training set to update our classifier. This Iterative approach of sample selection and updating the classifier helps in building an accurate classifier with less training samples. This process is independent of classifiers used. In this paper we are implementing pool based active learning with two different types of classifiers one is conventional machine learning algorithm Linear Support Vector Classifier and other is a transformer based model Bidirectional Encoder Representations from Transformers(BERT). We were able to annotate the entire dataset using 20% of samples from the dataset for training, This 20% of samples was annotated manually.

Introduction:- 
In Natural Language Processing, sentiment analysis is used to classify text according to whether it expresses an opinion and whether that opinion is positive, negative, or neutral. Text can also be categorised using advanced sentiment analysis based on its emotional state, such as angry, happy, or sad. It is frequently utilised in qualitative data analysis of user feedback, reviews, and social media posts, as well as customer experience research and user research. This application requires a huge amount of correctly annotated dataset to build an accurate model. Advancement in social media platforms like Twitter, Facebook, Reddit Lead to generation of huge volumes of textual data. The process of annotating these text samples becomes tedious and time consuming, therefore there is a requirement to develop an automatic approach to annotate these text data. In this paper we are using pool based active machine learning with two different types of classifiers. The classifiers used with this active learning algorithm include linear Support Vector Classifier which happens to be a simple conventional machine learning algorithm and the next classifier includes BERT a complex transformer based machine learning technique for NLP. A problem we want to solve is labelling unlabelled data and training models with the least of samples. In this paper to simulate the reliability of the algorithm we are testing the algorithm on two highly imbalanced dataset. 
The three main pillars in our attempt are pretrained models, Sentiment analysis and active learning.
Pretrained models 
Classification problems in NLP such as sentiment analysis can be broken down into two main processes: first is to understand the context behind the text and second based on the understanding to classify it to its respective class that it belongs to. The process of training the model to understand the text requires a huge training set and is time consuming. With the help of transfer learning we can fine tune the pre existing advanced BERT model and add additional neural network layers as per the task. Here the first hurdle of feature extractor is available in Hugging face community hub.
Sentiment analysis 
Identifying the perspective or emotion behind a situation is the goal of sentiment analysis, as the name suggests. It basically means to look at a piece of writing, speech, or other form of communication and figure out what it means to feel or mean. The machine does not view text as we humans do hence we have to represent text in the form of vectors which the model uses to classify. In preprocessing we convert the given text into vectors. Hugging face community also has embedding and tokenizer libraries for BERT that can convert the text into vectors. 
Active Learning
A subset of machine learning known as active learning involves the model suggesting the dataset's most relevant data points. The goal is to use model involvement to select these samples from the dataset rather than passive model learning. There are different ways to get these desired samples with least confident sampling, marginal sampling, entropy sampling. This is a significant part of methodology because we do not want random samples to be thrown in model training, important sample selection can be done using favourable strategy.

Related Work:- 
Let us look at the work related to active learning [1]. The work started with the proposal of a pool-based active learning model having an incremental decision tree as the basic classifier. There was also a method proposed in the removal of ambiguity in the unsampled pool. The best accuracy that was found was 92.50. In the next paper [2] active learning was performed on an imbalanced data and produced an efficient solution based on an extreme learning machine classification model. A detailed discussion on factors affecting active learning due to class imbalance was specified here. In this paper [3] active learning was implemented on the text dataset for Arabic text categorization. The dataset consisted of Arabic text for its labelling. The Support Vector machine was used for classification. Active learning on NADA Arabic [4] news data set. With this news dataset an accuracy of 99 percent was obtained when just 17.8 percent of samples were trained, active learning was proposed with a Jaccard similarity classifier. When coming to labelling multiple classes using active learning it is better to go by one versus all (OVA). OVA has always seen to be performing better in multi-class classification. An OVA-based decision tree was trained on 20 data streams and was seen performing better than pre existing models. The weight updation and training were found to be faster and the accuracy of classification was also higher. Drawbacks of OVA is not being able to handle concept change, classification accuracy gets affected by a dataset having an imbalanced class label.

Datasets and Preprocessing: 
To test the reliability and accuracy of the active learning algorithm. The dataset selected has unequal class distribution because in real life the data acquired may not have an evenly distributed class. Two dataset used include Financial News Dataset and Hugging face Emotion dataset. The properties of both the datasets are given in table 3.1 and 3.2.
Preprocessing: We are using two different types of classifiers, one is statical based and the other one is a transformer based NLP model. In the statistical method we calculate TF-IDF(Term Frequency-Inverse Document Frequency) based preprocessing after removing all the stopwords and punctuations. After preprocessing we get a vector later which we can use our desired classifier in this paper we will be using Support Vector Classifier. In the BERT classifier we used the pretrained tokenizer and word embedding. The hugging face community includes various tokenizers to convert texts into vectors this pretrained tokenizer is being used.

3.1: Financial News Dataset Properties
Name of the dataset : Financial News Dataset
Total Number of samples: 4846
class
Number of samples in each class
Example 
negative
604
The international electronic industry company Elcoteq has laid off tens of employees from its Tallinn facility ; contrary to earlier layoffs the company contracted the ranks of its office workers , the daily Postimees reported .
neutral
2879
Technopolis plans to develop in stages an area of no less than 100,000 square metres in order to host companies working in computer technologies and telecommunications , the statement said .
positive
1363
In the third quarter of 2010 , net sales increased by 5.2 % to EUR 205.5 mn , and operating profit by 34.9 % to EUR 23.5 mn .



3.2: Hugging face emotion dataset
Name of the dataset : Hugging face emotion dataset
Total Number of samples: 20,000
class
Number of samples in each class
Example 
sadness
5797
I’m feeling rather rotten so I'm not very ambitious right now


anger
2709
I felt anger when at the end of a telephone call
love
1641
I want each of you to feel my gentle embrace
joy
6761
I feel a little mellow today
fear
2373
I pay attention it deepens into a feeling of being invaded and helpless


surprise
719
i feel shocked and sad at the fact that there are so many sick people






Methodology : 
The block diagram of active learning given in fig 3.1 There are various algorithms existing based on active learning, but this is the basic flow diagram for pool based active learning. From the given unlabelled dataset we first select a few training samples randomly and annotate those selected samples. These annotated samples are used to train our model. The trained model is later tested on the remaining unlabelled batch, from this unlabelled batch some samples are selected based on a query strategy defined by the user. This process of annotation, training and sample selection is done for multiple iterations till it reaches a stopping condition. The stopping condition that we kept in our algorithm is when our query strategy does not select any new sample to annotate.


Fig. 3.1
Query strategy:
The strategy with which we decide if the sample should be selected to be annotated manually and added to our training pool is known as query strategy. Least confident sampling, marginal sampling, entropy sampling are the three main types of query strategy here we use a different approach to select our sample. In this paper we are using a maximum conflict approach. Here from the final output softmax layer we get the probabilities of each class. The class having the highest probability is selected and its probability is later compared to other class probability based on the similarity the score is defined. 
The similarity is calculated using the equation 4.2 
Similarity Index = 1 - abs(n1 - n2) / (n1 + n2) ----------- eq 4.2

Example :- Let us assume the model is trained with dataset having three classes:
Output matrix generated :-  [0.987, 0.980 , 0.5 ]
From the output matrix the the maximum number is selected in this case it is 0.987 which is then later compared with the remaining elements and calculates a similarity index the equation used to calculate the similarity index is given by 
Similarity Index = 1 - abs(n1 - n2) / (n1 + n2)
Based on this similarity index we find out if the sample is conflicting or not. 

Results:- 
In this paper we are comparing the result between two approaches. One where samples
are selected randomly to train the classifier and other approach where we use active
learning to select the samples.
The below Tables consists of Average F1 scores as even in active learning the initial
samples we select are random because of which the accuracy, F1 score and number
of samples selected are also affected therefore the average is taken to the number of
times this process is repeated.

Training without using active learning algorithm
In this process we select a defined number of samples randomly To train our classifier
which will later classify the rest of the samples. In the below tables 4.1,4.2,4.3,4.4 we have
used the training samples to train the active learning model and tested the model on
the remaining samples.
For example: We use 750 samples out of 4800 samples for training the classifier
The remaining samples 4050 will be used for testing.
The below table 4.1 and 4.2 contains the F1 scores when classifier trained with
conventional machine learning classifier - Linear SVC:



The below table 4.3 and 4.4 contains the F1 scores when classifier trained with
Transformer based classifier - BERT:


From the above table we can observe that selecting more samples does not necessarily
improve the accuracy. This happens because there is a class imbalance in the
dataset, which is why we cannot select the same number of samples each time. Selecting
the same number of samples from each class affects the F1 score of that particular
class.

Training with active learning algorithm
In the previous method we selected equal number of samples from each class even if
The class distribution is not even. Doing that will affect the overall accuracy of the
model. This approach with active learning we will be selecting samples and making a
training pool. When the training pool is ready this pool is used to train the classifier.
Once the classifier is trained the classifier model classifies the testing samples and
generates the F1 scores.


From the above Tables we can notice that changing the classifier model may increase
the accuracy and F-1 score but does not necessarily mean that it can improve
the over all algorithm. There is a need to change the query strategy and a better
procedure to select a better initial samples for training.















Additional Content :
Abstract :- Due to the advancement of social media platforms, like Twitter, Facebook, Reddit etc. and the increase in its number of users. Led to the generation of huge amounts of textual data. These Textual data can be used in areas such as sentiment analysis. The process of annotating these textual data is time consuming and tedious in nature, Hence we use this approach of pool based active learning with Natural Language Processing. In this paper we are implementing an active learning model with BERT as a classifier, comparing with other machine learning algorithms as a classifier. 

In many machine learning and deep learning applications there is a requirement of annotated data when it comes to supervised learning. The performance of the model is highly dependent on the degree of accuracy in which the data set is annotated. This process of data labelling can be time consuming and tedious in nature. This labour intensive process can be made more efficient and mitigate human interaction with the help of active learning. Active Learning has the potential of making the process of data annotation quicker, efficient and reducing any human interaction for annotation. With the advancement in the field of Natural language processing it is possible to fine tune pre-trained transformer based models like BERT(Bidirectional Encoder Representations from Transformers) that are capable of getting proper context from texts which makes it perfect for any NLP task. In this paper we are implementing pool based active learning using different classification algorithms to annotate text sentiments. 

Introduction:- After the advancement of social media platforms, such as Twitter, Facebook, Reddit etc. and the increase in its number of users. That resulted in availability of a large amount of textual data, annotating these samples becomes a tedious and time consuming task. With the help of active learning this process can be done efficiently with less human interaction.   
In the field of Natural Language Processing (NLP), We have seen development of various types of Deep learning architectures and improvement in Natural Language Understanding (NLU). When it comes to sentiment analysis such deep learning architectures can be used as a classifier with active learning algorithms.
When it comes to deep learning models there is a requirement of a huge number of training samples to make a better model. But now due to the availability and easy access to open source pre-trained NLP models it is possible to build a fairly accurate model with less number of training samples. This transfer learning approach gives us an edge while implementing any active learning algorithm. In active learning we do not select any random set of samples to train our classifier but samples are selected based on a query which decides if those samples have to be added in the pool or not. 

Research Gap:- In recent times when it comes to Natural Language Processing there are many models currently available such as RNN, LSTM , GRU etc. But these architectures are not the best when it comes to understanding the context behind the text. In active learning using such advanced encoder based architecture with transfer learning will give an edge over the above mentioned architectures. The use of transfer learning also makes it possible to use a training batch with less number of training samples. 



Pool Based active learning: 
In an active learning problem the machine selects some unlabelled samples which it then queries the oracle for a label. 
Pool based active learning is one such approach where the machine has access to a large amount of unlabelled data which can also be called an unlabelled pool. The classifier then selects samples based on a certain criteria known as query, after which these selected unlabeled samples are labelled and added to the training pool. 

Initial step manually Labelling samples:
In this step we randomly select equal number samples from each class and annotate these samples manually. This is the first stage of training, here the samples are not selected with the help of a query. These samples are added to the training pool which will be modified iteratively. 

Training the classifier:- 
The training is done with the samples in the training pool. After training the model, The model later classifies the remaining unlabelled samples in the unlabelled pool and generates a confidence score matrix. Query strategy is then applied to the score matrix based on which the sample is decided whether to be added in the training pool or not.

Example :- Let us assume the model is trained with dataset having three classes:
Output matrix generated :-  [0.987, 0.980 , 0.5 ]
From the output matrix the the maximum number is selected in this case it is 0.987 which is then later compared with the remaining elements and calculates a similarity index the equation used to calculate the similarity index is given by 
Similarity Index = 1 - abs(n1 - n2) / (n1 + n2)
Based on this similarity index we find out if the sample is conflicting or not. 

















Result :- 
Training without using active learning algorithm
In this process we select a defined number of samples randomly.

conventional machine learning classifier - Linear SVC:
Hugging face emotion dataset 


Financial News headline dataset 












Transformer based classifier - BERT:
Hugging face emotion dataset 


Financial News headline dataset 



Training with using active learning algorithm
Active Learning with conventional machine learning classifier - Linear SVC:
Hugging face emotion dataset 
20000 Total samples 4500 training samples, Annotated 15500 samples with an accuracy of 85%
F1 scores
Anger 90, fear 80, joy 86, love 71 , sadness 88, surprise 71

Financial News headline dataset 
4846 Total samples 1349 training samples, annotated 3497 samples with an accuracy of 73%
F1 scores
Negative:  0.62 Neutral: 0.82 , Positive: 0.52


Active Learning with Transformer based classifier - BERT:
Hugging face emotion dataset 
20000 Total samples 3800 training samples annotated 16200 samples with an accuracy of 92%
F1 scores
Anger 95, fear 87, joy 93, love 83 , sadness 96, surprise 78

Financial News headline dataset 
4846 Total samples  800 training samples, annotated 4046 samples with an accuracy of 80%
F1 scores
Negative:  0.83 Neutral:0.81  , Positive: 0.61









Extras
Objective 1 :- To develop a robust active learning algorithm capable enough to annotate 80% or more samples accurately with minimum human effort for annotation.

Objective 2 :-  Optimising the algorithm to make it more time efficient and reducing the time taken for the classifier to be trained that is used in the active learning algorithm.

Objective 3:- Building a generalised and flexible active algorithm that can be used to annotate images by plugging in another suited classifier model. 
