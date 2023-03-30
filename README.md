<h1 align="center">Image Caption Generation</h1>
<h4 align="center">For course fulfillment of Fuse Micro-degree Deep Learning</h4>

<h3 align="center"> Sushmita Poudel &nbsp;&nbsp;•&nbsp;&nbsp;Sonika Aacharya &nbsp;&nbsp;•&nbsp;&nbsp; Shailesh Ashikari &nbsp;&nbsp;•&nbsp;&nbsp; Shulav Karki </h3>

---

Jan 18, 2022

# Introduction
Generating a descriptive caption for an image using proper English language is a difficult task for a machine as it requires an algorithm to comprehend not only the content of an image but also generate language that accurately describes it. Our project aims to tackle this challenge by developing a generative model that integrates recent advances in both Computer Vision and Natural Language Processing. The outcome of this project has various potential applications, such as automating image tagging and metadata generation, improving social media marketing, and aiding visually impaired individuals in comprehending the context of images.

## Problem Statement
Image Captioning task requires an algorithm to not only understand the content of the image, but also to generate language that connects to its interpretation. In this particular project we have designed an application to base the caption generator for instagram posts. A simple caption generator derives literal interpretation of the image while this project generates the caption that is more instagram-like.

### Dataset Used:
The dataset - Flickr 8k used in this project is collected from kaggle repository. The dataset consists of around 8000 images that are each paired with five different captions which provide a clear description of each salient features of the image. The images were taken from six different Flickr groups, and tend not to contain any well known people or location but were manually selected to depict a variety of scenes and situations.

Find dataset at : https://www.kaggle.com/datasets/adityajn105/flickr8k

### Model Architecture

- **VGG16 Classification model**: 
Since our model takes a similar approach to VGG16, a convolutional neural network trained on 1.2 million images to classify 1000 different categories of images, we used it to extract major features from the images. We dropped the very last layer of the network and used the rest.
![vgg16](https://user-images.githubusercontent.com/66173400/228822590-dcf6a90f-bd46-4bfb-9669-bb6d5f9047b3.jpg)
<h5 align="center">Fig 1: VGG-16 Architecture</h5>

- **LSTM- based Sentence Generator**: 
Our caption generator is based on LSTM, a special kind of Recurrent Neural Network that is capable of learning long-term dependencies.<br>
---> Input layer with input length = 35 ( max_length after tokenization)    <br>
---> Embedding layer input = 35,<br> 
---> output = 256<br>
---> Dropout layer with alpha = 0.4 <br>
---> LSTM layer with input = 256 

![Screenshot from 2023-03-30 17-30-40](https://user-images.githubusercontent.com/66173400/228826321-2b21a767-b711-491b-903e-cd17916b2fcf.png)
<h5 align="center">Fig 2: Word embedding and LSTM layers architecture</h5>

- **Preprocessing and parameters**
In our approach for feature extraction, we resized all images to 224 x 224 without performing any image augmentation. This was due to the fact that we had a sufficiently large dataset of over 8,000 images that we believed would allow our model to generalize well and produce accurate results. For text cleaning, we applied lowercase conversion, removal of special characters (excluding emojis), and trailing spaces. We used the default configuration of the VGG-16 network for feature extraction and employed the Keras text preprocessing Tokenizer to tokenize the captions. The data was split into a 90-10 ratio and trained for 15 epochs with a batch size of 64.

### FINAL MODEL
Our ultimate neural network model utilizes features extracted from VGG-16, which are initially shaped as (,4096) but are then compressed to (,256), and features from an LSTM network, which have an initial shape of (,35) but are expanded to (,256). The model's final neural network head has an output of 8485 words, which is equivalent to the size of our vocabulary. Through a softmax activated output layer, the model can determine the likelihood of occurrence for each word in generating a descriptive caption for an image.
![Screenshot from 2023-03-30 17-30-40 (1)](https://user-images.githubusercontent.com/66173400/228826386-18466ee5-7e81-4718-b192-95ea82088d7b.png)
<h5 align="center">Fig 3.: Final architecture of model with VGG-16 , LSTM and Dense layers</h5>

### EVALUATION METHOD
To evaluate the performance of our model on the test set, we employed two techniques.
* **BLEU (BiLingual Evaluation Understudy)** score, a metric for automatically evaluating machine-translated text. It is calculated for each individual translated segment - by comparing the machine translated text to a good quality reference text: “the closer a machine translation is to a professional human translation, the better it is”.
The higher the Bleu score, the better the model. A score of 0.7 or 0.8 is considered the best you can achieve. A BLEU score of 1 means that the candidate sentence perfectly matches one of the reference sentences.

* Also, the model performance can be evaluated in a subjective way where each generated caption will be evaluated manually, similar to the concept of Amazon Mechanical Turk where each image was rated by 2 workers and we performed bootstrapping for variance analysis, and scores will be assigned in the scale of 0 to 4 on the usefulness of each description given the image.

### RESULT AND FUTURE WORK
We have developed an end-to-end neural network model that utilizes a convolutional neural network to encode an image into a vector representation, followed by a recurrent neural network that generates a descriptive caption for the image using proper English language. The model achieved an average BLEU score of 0.567247, indicating its effectiveness in accurately generating captions for images.

Expanding on our image to caption generator model, we can further develop a caption to speech converter, which would enable visually impaired individuals to comprehend the content of images in public places. Additionally, to improve the model's performance, we could attempt to train it for more epochs, although we were unable to do so this time due to RAM and GPU limitations. Furthermore, as our project was based on a labeled dataset, we could explore the model's performance on an unlabeled dataset in the future, applying unsupervised learning techniques to the architecture.

#### REFERENCES
VGG-16 CNN for feature extraction : https://www.geeksforgeeks.org/vgg-16-cnn-model/ <br>
Tokenizer: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer
