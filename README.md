# cs7643-deep-learning-solved
**TO GET THIS SOLUTION VISIT:** [CS7643: Deep Learning Solved](https://mantutor.com/product/cs7643-deep-learning-solved-6/)


---

**For Custom/Order Solutions:** **Email:** mantutorcodes@gmail.com  

*We deliver quick, professional, and affordable assignment help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;98392&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;3&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (3 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;CS7643: Deep Learning Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (3 votes)    </div>
    </div>
Assignment 1

• It is your responsibility to make sure that all code and other deliverables are in the correct format and that your submission compiles and runs. We will not manually check your code (this is not feasible given the class size). Thus, non-runnable code in our test environment will directly lead to a score of 0. Also, your entire programming parts will NOT be graded and given 0 score if your code prints out anything that is not asked in each question.

Overview

Deep Neural Networks are becoming more and more popular and widely applied to many ML-related domains. In this assignment, you will complete a simple pipeline of training neural networks to recognize MNIST Handwritten Digits: http://yann.lecun.com/exdb/mnist/. You’ll implement two neural network architectures along with the code to load data, train and optimize these networks. You will also run different experiments on your model to complete a short report. Be sure to use the template of report we give to you and fill in your information on the first page.

The main.py contains the major logic of this assignment. You can execute it by invoking the following command where the yaml file contains all the hyper-parameters.

$ python main.py –config configs/&lt;name_of_config_file&gt;.yaml

There are three pre-defined config files under ./configs. Two of them are default hyper-parameters for models that you will implement in the assignment (Softmax Regression and 2-layer MLP). The correctness of your implementation is partially judged by the model performance on these default hyper-parameters; therefore, do NOT modify values in these config files. The third config file, config_exp.yaml, is used for your hyper-parameter tuning experiments (details in Section 5) and you are free to modify values of the hyper-parameters in this file.

The script trains a model with the number of epochs specified in the config file. At the end of each epoch, the script evaluates the model on the validation set. After the training completes, the script finally evaluates the best model on the test data.

Python and dependencies

In this assignment, we will work with Python 3. If you do not have a python distribution installed yet, we recommend installing Anaconda: https://www.anaconda.com/ (or miniconda) with Python 3. We provide environment.yaml which contains a list of libraries needed to set environment for this assignment. You can use it to create a copy of conda environment. Refer to the users’ manual: https://docs.conda.io/projects/conda/en/latest/userguide/tasks/manage-environments.html for more details.

$ conda env create -f environment.yaml

If you already have your own Python development environment, please refer to this file to find necessary libraries, which is used to set the same coding/grading environment.

Code Test

There are two ways(steps) that you can test your implementation:

1. Python Unit Tests: Some public unit tests are provided in the tests/ in the assignment repository. You can test each part of your implementation with these test cases by:

$ python -m unittest tests.&lt;name_of_tests&gt;

However, passing all local tests neither means your code is free of bugs nor guarantees that you will receive full credits for the coding section. Your code will be graded by GradeScope Autograder(see below for more details). There will be additional tests on GradeScope which does not present in your local unit tests.

1 Data Loading

Data loading is the very first step of any machine learning pipelines. First, you should download the MNIST dataset with our provided script under

./data by:

$ cd data

$ sh get_data . sh

$ cd . . /

Microsoft Windows 10 Only C:assignmentfolder&gt; cd data C: assignmentfolder data&gt; get_data . bat C: assignmentfolder data&gt; cd . .

The script downloads MNIST data (mnist_train.csv and mnist_test.csv) to the ./data folder.

1.1 Data Preparation

To avoid the choice of hyper-parameters overfits the training data, it is a common practice to split the training dataset into the actual training data and validation data and perform hyper-parameter tuning based on results on validation data. Additionally, in deep learning, training data is often forwarded to models in batches for faster training time and noise reduction.

In our pipeline, we first load the entire MNIST data into the system, followed by a training/validation split on the training set. We simply use the first 80% of the training set as our training data and use the rest training set as our validation data. We also want to organize our data (training, validation, and test) in batches and use different combination of batches in different epochs for training data. Therefore, your tasks are as follows: (a) follow the instruction in code to complete load_mnist_trainval in ./utils.py for training/validation split (b) follow the instruction in code to complete generate_batched_data in ./utils.py to organize data in batches

You can test your data loading code by running:

$ python -m unittest tests.test_loading

2 Model Implementation

You will now implement two networks from scratch: a simple softmax regression and a two-layer multi-layer perceptron (MLP). Definitions of these classes can be found in ./models.

Weights of each model will be randomly initialized upon construction and stored in a weight dictionary. Meanwhile, a corresponding gradient dictionary is also created and initialized to zeros. Each model only has one public method called forward, which takes input of batched data and corresponding labels and returns the loss and accuracy of the batch. Meanwhile, it computes gradients of all weights of the model (even though the method is called forward!) based on the training batch.

2.1 Utility Function

There are a few useful methods defined in ./_base_network.py that can be shared by both models. Your first task is to implement them based on instructions in _base_network.py:

(a) Activation Functions. There are two activation functions needed for this assignment: ReLU and Sigmoid. Implement both functions as well as their derivatives in ./_base_network.py (i.e, sigmoid, sigmoid_dev, ReLU, and ReLU_dev). Test your methods with:

$ python -m unittest tests.test_activation

$ python -m unittest tests.test_loss

2.2 Model Implementation

You will implement the training processes of a simple Softmax Regression and a two-layer MLP in this section. The Softmax Regression is composed by a fully-connected layer followed by a ReLU activation. The two-layer MLP is composed by two fully-connected layers with a Sigmoid Activation in between. Note that the Sofmax Regression model has no bias terms, while the two-layer MLP model does use biases. Also, don’t forget the softmax function before computing your loss!

(a) Implement the forward method in softmax_regression.py as well as two_layer_nn.py. If the mode argument is train, compute gradients of weights and store the gradients in the gradient dictionary. Otherwise, simply return the loss and accuracy. Test:

$ python -m unittest tests.test_network

3 Optimizer

We will use an optimizer to update weights of models. An optimizer is initialized with a specific learning rate and a regularization coefficients. Before updating model weights, the optimizer applies L2 regularization on the model:

(1)

where J is the overall loss and LCE is the Cross-Entropy loss computed between predictions and labels.

You will also implement an vanilla SGD optimizer.The update rule is as follows:

θt+1 = θt − η∇θJ(θ) (2)

where θ is the model parameter, η stands for learning rate and the ∇ term corresponds to the gradient of the parameter. In summary, your tasks are as follows:

(b) Implement the update method in sgd.py based on the discussion of

update rule above.

Test your optimizer by running:

$ python -m unittest tests.test_training

4 Visualization

It is always a good practice to monitor the training process by monitoring the learning curves. Our training method in main.py stores averaged loss and accuracy of the model on both training and validation data at the end of each epoch. Your task is to plot the learning curves by leveraging these values. A sample plot of learning curves can be found in Figure 1.

Figure 1: Example plot of learning curves

5 Experiments

Now, you have completed the entire training process. It’s time to play with your model a little. You will use your implementation of the two-layer MLP for this section. There are different combinations of your hyper-parameters specified in the report template and your tasks are to tune those parameters and report your observations by answering questions in the report template. We provide a default config file config_exp.yaml in ./configs. When tuning a specific hyper-parameter (e.g, the learning rate), please leave all other hyper-parameters as-is in the default config file.

(a) You will try out different values of learning rates and report your observations in the report file.

(b) You will try out different values of regularization coefficients and report your observations in the report file.

(c) You will try your best to tune the hyper-parameters for best accuracy.

6 Deliverables

6.1 Coding

To submit your code to Gradescope, you will need to submit a zip file containing all your codes in structure. For your convenience, we provide a handy script for you.

Simply run

$ bash collect_submission . sh or if running Microsoft Windows 10

C:assignmentfolder&gt;collect_submission . bat then upload assignment_1_submission.zip to Gradescope.

6.2 Writeup

You will also need to submit a report summarizing your experimental results and findings as specified in Section 5. Again, we provide a starting template for you and your task is just to answer each question in the template. For whichever quesitons asking for plots, please include plots from all your experiments.

Note: Explanations should go into why things work the way they do with proper deep learning theory. For example, with hyperparameter tuning you should explain the reasoning behind your choices and what behavior you expected. If you need more than one slide for a question, you are free to create new slides right after the given one.

You will need to export your report in pdf format and submit to Gradescope. You should combine your answers to the theory questions with your report into one pdf and submit it to the ”Assignment 1 Writeup” assignment in Gradescope. When submitting to Gradescope, make sure you select ALL corresponding slides for each question. Failing to do so will result in -1 point for each incorrectly tagged question, with future assignments having a more severe penalty.
