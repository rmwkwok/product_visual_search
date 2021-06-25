# Introduction

Visual search is the most interesting entry to computer vision because it connects images. Don't we always say "A picture is worth a thousand words"? Wouldn't it be amazing to see in one day a search engine that will pick some "interesting" aspects in your photo and show you some relevant further readings?

[This product visual search competition](https://eval.ai/web/challenges/challenge-page/888/overview) (Yuan et al.) was not exactly that idea but it was like a chance to get hands dirty. Its goal was to, given a picture of a product, look for other pictures of the same product, and those other pictures could just be taken from different angles. 

It certainly has its own interesting applications as well, and as far as I could find, giant online retailer sites like Amazon and Alibaba may group user's search result by sellers who sell the same product. Even it is not about presenting search result, its ability to correlate product information by photos enables many potentials such as price-matching. 

Well, enough thinking!

# Approach

(Be careful, I only got the fifth place and did not win the first place in the competition. Though the time limitation had something to do with my final score, frankly I did not know that state-of-the-art approach for this problem and the following was only my understanding and about my journey.)

I was really very carefully approaching it because I had never done this. That's why, instead of training a brand new model, I started with a pre-trained EfficientNetB0 which was a state-of-the-art (Tan & Le). From it I gradually opened more and more sections of the network for training, in the hope to slowly adapt the EfficientNetB0 to my new dataset.

The dataset offered by the organizer, as the competition site mentioned, had over 1 million product images, and having had a glance of them, they looked pretty normal, so in my image augmentation, I used flipping and rotations most of the time. There were two reasons for having the augmentions. Firstly, this was what the competition was about - being able to recognize the same product from different ways of presenting it in pictures, so it would be a good idea if the tuned EfficientNetB0 could correctly categorize the original and the augmented versions in the same group. Thus, categorical cross-entropy was the loss function for fine tuning the network. Secondly, which is also what I am going to talk about next, is I needed them for training my Siamese network.

The idea of the Siamese network (Koch et al.) in this context was that, the two pictures of the same origin would result in similar embeddings after going through the network, and certainly, those with different origins would have different embeddings, and the training of such model was achievable by using the semi-hard triplet loss (Schroff et al.). In the most ideal and extreme case (which I could only try to get close to), embeddings of the same products had a cosine-similarity of 1 while those of different products -1.

To connect all the dots in this section, I first gradually fine-tuned the pre-trained EfficientNetB0. Then I passed all the pictures in my dataset through it to produce embeddings for each picture. With the initial embeddings, I trained the Siamese model which effectively acted as a projection network which projected the initial embeddings to some final ones that carry the similarity nature described above. To illustate the steps of training, I had the following section.

# Training steps

![architecture](https://github.com/rmwkwok/product_visual_search/blob/1cb7370dbc42042bd450e48de8b61ac21c6bde43/images/architecture.png)

It was divided into 4 phases. 

- Phase 1: A new top layer was trained with a learning rate (lr) of 1e-2 for 2 epochs
- Phase 2: Layers from block7a of the EfficientNetB0 onwards were trained with a lr of 1e-4 for 1 epoch
- Phase 3: The whole model was trained with a lr of 1e-5 for 2 epochs

Compared with before Phase 1, the leader board (LB) score after Phase 3 increased from 0.342 to 0.44.

- Phase 4: Output from the last Batch Normalization Layer of the EfficientNetB0 was used as input for training the Siamese part for 4 epochs. In the first 2 epochs, a lr and a triplet loss margin of 1e-5 and 0.5 were used, followed by 1e-5 and 0.75 in the 3rd epoch, and 1e-4 and 0.75 in the last one. 

At the end of these epochs of Phase 4, the LB scores became 0.449, 0.454, 0.459, and 0.463 respectively, which was picking up an increasing trend.

You might be wondering how I had tuned the hyper-parameters and why I did not train for more epochs. On the one hand, I was training the model against the LB score (which was the "ground truth"), and I onlt reported the best numbers here. On the other hand, I had too many things to try out in a limited period of time, so the LB scores of my phase 4 could have been further improved if I had allocated more time on it.

# Finally

This concluded, and if you are interested, below is a summary of my LB score journey. I can do better next time.

![LB scores](https://github.com/rmwkwok/product_visual_search/blob/1cb7370dbc42042bd450e48de8b61ac21c6bde43/images/journey.png)

# Acknowledgement

I thank the organizer of the competition. With this target, I tried out technologies such as GPU/TPU/Cloud, built my data pipeline for training, and tensorflow functions using the TF library. With the leader board, I got to test my ideas, keeping some and dropping most, which was very invaluable experience.

# About the code

It is not 100% of the code I used for the competition. I skipped the part for generating the TFRecords from raw data, because the raw dataset was not public. I also skipped the part for creating the submission file and the part for visualizing the dataset. My purpose of this repository is to show how I did the training, and perhaps the most informative file is actually this readme.md.

# Reference

G. Koch, R. Zemel & R. Salakhutdinov. Siamese neural networks for one-shot image recognition. Retrieved from https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf

F. Schroff, D. Kalenichenko & J. Philbin. FaceNet: a unified embedding for face recognition and clustering. Retrieved from https://arxiv.org/abs/1503.03832

M. Tan & Q. V. Le. EfficientNet: rethinking model scaling for convolutional neural networks. Retrieved from https://arxiv.org/abs/1905.11946

J. Yuan, A. Chiang, W. Tang & T. Haro: eProduct: a million-scale visual search benchmark to address product recognition challenges. Retrieved from https://drive.google.com/file/d/1DkN5xdrE7rytEmHyKmrHQI36JIXjtIhM/view?usp=sharing

