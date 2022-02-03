## Audio+Video Emotion Recognition

### Introduction

Traditional audiovisual fusion systems consist of two stages, feature extraction from the image and audio signals and combination of the features for joint classiﬁcation . Although decades of research in acoustic speech recognition have resulted in a standard set of audio features, there is not a standard set of visual features yet. This issue has been recently addressed by the introduction of deep learning in this ﬁeld.

 As shown in the figure below:



#### What was already done in literature?

1. Deep audio-visual speech recognition
   This article compares the performance of TM-Seq2seq and TM-CTC in audiovisual recognition. Through experiments, it is found that even under noisy conditions, the TM-CTC audio-visual model performs better than the audio-only or visual-only model; When using the audio-visual TM-CTC model, the word error rate was reduced from 10.1% for audio only to 8.2%.

2. Audio-visual speech recognition with a hybrid ctc/attention architecture

   In this work, Present a joint CTC/attention hybrid architecture for audio-visual speech recognition. Results on the LRS2 database show that the audio-visual model signiﬁcantly outperforms the audio-only model especially at high levels of noise and also achieves the new state-of-the-art performance on this dataset. 
   
3. Learning Affective Features With a Hybrid Deep Model for Audio–Visual Emotion Recognition

   This paper presents a new method for audio-visual emotion recognition with a hybrid deep learning framework integrating CNN, 3D-CNN and DBN. The outputs of audio and visual networks are connected with a deep DBN model to fuse audio and visual cues.Experimental results on the RML, eNTERFACE05, and BAUM-1s datasets show that our hybrid deep learning model jointly learns a discriminative audio-visual feature representation, which performs better than previous hand-crafted features and fusion methods on emotion recognition tasks.

3. A Combined Rule-Based & Machine Learning Audio-Visual Emotion Recognition Approach

   This paper propose a different set of rules to recognize a different set of six emotions (anger, happy, sad, disgust, surprise and fear). These are the six universal emotional states proposed by the psychologist Paul Ekman.

   This paper proposes an audio-visual emotion recognition system that uses a mixture of rule-based and machine learning approaches to improve the recognition efficacy.
   
5. Emotion recognition using deep learning approach from audio-visual emotional big data

   This paper proposes an audio-visual emotion recognition system using a deep network to extract features and another deep network to fuse the features.

   The contributions of this paper are (i) the proposed system is trained using Big Data of emotion and, therefore, the deep networks are trained well, (ii) the use of layers, one layer for gender separation and another layer for emotion classification, of an extreme learning machine (ELM) during fusion; this increases the accuracy of the system, (iii) the use of a two dimensional convolutional neural network (CNN) for audio signals and a three dimensional CNN for video signals in the proposed system; a sophisticated technique to select a key frame is also proposed, and (iv) the use of the local binary pattern (LBP) image and the interlaced derivative pattern (IDP) image together with the gray-scale image of key frames in the three dimensional CNN; in this way, different informative patterns of key frames are given to the CNN for feature extraction.

#### What was possible to do?

1. Deep audio-visual speech recognition

2. Audio-visual speech recognition with a hybrid ctc/attention architecture

   It is possible to investigate in future work an adaptive fusion mechanism which learns to weight each modality based on the noise levels.
   
3. Learning Affective Features With a Hybrid Deep Model for Audio–Visual Emotion Recognition

   1.  It is thus meaningful to investigate how to reduce the network parameters of deep models, e.g., deep compression , to achieve real-time emotion recognition with a deep model.
   2.  More robust face detectors and models will be studied in our future work.
   3.  End-to-end learning and recognition strategies would also be investigated in our future work.
   4.  Will investigate the performance of CNN+LSTM for facial expression recognition in our future work.
   
4. A Combined Rule-Based & Machine Learning Audio-Visual Emotion Recognition Approach

   future work will look at practical problems associated with this application such as the length of the window segment and percentage of overlap for the audio path, and the number of frames within a block for the visual path.
   
5. Emotion recognition using deep learning approach from audio-visual emotional big data

   The proposed system can be integrated in to any emotion-aware intelligent systems for a better service to the users or customers. Using edge technology, the weights of the deep network parameters can easily be stored for fast processing.

#### How the speech recognition was improved based on both audio and video input?

1. Deep audio-visual speech recognition/

2. Audio-visual speech recognition with a hybrid ctc/attention architecture

#### **reference**

1. Petridis, S., Stafylakis, T., Ma, P., Tzimiropoulos, G., & Pantic, M. (2018, December). Audio-visual speech recognition with a hybrid ctc/attention architecture. In *2018 IEEE Spoken Language Technology Workshop (SLT)* (pp. 513-520). IEEE.
2. Afouras, T., Chung, J. S., Senior, A., Vinyals, O., & Zisserman, A. (2018). Deep audio-visual speech recognition. *IEEE transactions on pattern analysis and machine intelligence*.
2. Zhang S, Zhang S, Huang T, et al. Learning affective features with a hybrid deep model for audio–visual emotion recognition[J]. IEEE Transactions on Circuits and Systems for Video Technology, 2017, 28(10): 3030-3043.
2. Seng K P, Ang L M, Ooi C S. A combined rule-based & machine learning audio-visual emotion recognition approach[J]. IEEE Transactions on Affective Computing, 2016, 9(1): 3-13.
2. Hossain M S, Muhammad G. Emotion recognition using deep learning approach from audio–visual emotional big data[J]. Information Fusion, 2019, 49: 69-78.

### Algorithms

#### Transformer self-attention architecture

1. Sequence-to-sequence Transformer (TM-seq2seq)

   In this variant, separate attention heads are used for attending on the video and audio embeddings. In every decoder layer, the resulting video and audio contexts are concatenated over the channel dimension and propagated to the feedforward block. The attention mechanisms for both modalities receive as queries the output of the previous decoding layer (or the decoder input in the case of the ﬁrst layer). The decoder produces character probabilities which are directly matched to the ground truth labels and trained with a cross-entropy loss.

2. CTC Transformer (TM-CTC)

   The TM-CTC model concatenates the video and audio encodings and propagates the result through a stack of self-attention / feedforward blocks, same as the one used in the encoders. The outputs of the network are the CTC posterior probabilities for every input frame and the whole stack is trained with CTC loss.

3. Architecture

   <img src="E:\Users\DingYi\Desktop\1.27 report\Report_12.28.assets\image-20220126160546679.png" alt="image-20220126160546679" style="zoom: 80%;" />

   1. **Encoder Input:**

      Video: The input images are 224×224 pixels, sampled at 25 fps and contain the speaker's face. We crop a 112x112 patch covering the region around the mouth.

      Audio: For the acoustic representation we use 321-dimensional spectral magnitudes, computed with a 40ms window and 10ms hop-length, at a 16 kHz sample rate. Since the video is sampled at 25 fps (40 ms per frame), every video input frame corresponds to 4 acoustic feature frames. 

   2. **Decoder Output**:

      TM-Seq2seq: The decoder produces character probabilities which are directly matched to the ground truth labels and trained with a cross-entropy loss.

      TM-CTC: The outputs of the network are the CTC posterior probabilities for every input frame and the whole stack is trained with CTC loss. 

#### Hybrid CTC/Attention architecture

​	To map a set of input sequences such as audio or video streams to corresponding output sequences, we consider a hybrid CTC/attention architecture [20] in this paper. This architecture uses a typical encoder-decoder attention structure. A stack of Bidirectional Long Short Term Memory Networks (BLSTMs) is employed in the encoder to convert input streams x = (x1,...,xT ) into frame-wise hidden feature representations. These features are then consumed by a joint decoder including a recurrent neural network language model (RNN-LM), attention and CTC mechanisms to output a label sequence y = (y1,...,yL). To perform alignment between input frames and output characters, we use a location based attention mechanism, which takes into account both content and location information for selecting the next step in the input sequence [23].

<img src="E:\Users\DingYi\Desktop\1.27 report\Report_12.28.assets\image-20220126160559313.png" alt="image-20220126160559313" style="zoom: 80%;" />

<img src="E:\Users\DingYi\Desktop\1.27 report\Report_12.28.assets\image-20220126160607889.png" alt="image-20220126160607889" style="zoom: 80%;" />

#### Transformer structure

<img src="E:\Users\DingYi\Desktop\1.27 report\Report_12.28.assets\image-20220126160654378.png" alt="image-20220126160654378" style="zoom:50%;" />

Transformer also uses the classic Encoder and Decoder architecture, which is composed of Encoder and Decoder.

The structure of Encoder is composed of Multi-Head Self-Attention and position-wise feed-forward network. The input of Encoder is composed of the sum of Input Embedding and Positional Embedding.

The structure of Decoder is composed of Masked Multi-Head Self-Attention, Multi-Head Self-Attention and position-wise feed-forward network. The initial input of Decoder is obtained by summing Output Embedding and Positional Embedding.

The part of the Nx box on the left half of the above figure is the first layer of the Encoder, and the Encoder in the Transformer has 6 layers.

The part framed by Nx on the right half of the figure above is a layer of Decoder, and there are 6 layers of Decoder in Transformer.

<img src="E:\Users\DingYi\Desktop\1.27 report\Report_12.28.assets\image-20220126160714589.png" alt="image-20220126160714589" style="zoom:67%;" />

Encoder

<img src="E:\Users\DingYi\Desktop\1.27 report\Report_12.28.assets\image-20220126160726196.png" alt="image-20220126160726196" style="zoom:67%;" />

Encoder consists of 6 identical layers, each layer contains 2 parts:

- Multi-Head Self-Attention
- Position-Wise Feed-Forward Network (Fully Connected Layer)

Both parts have a residual connection (redidual connection), and then a Layer Normalization.

The input of Encoder consists of the sum of Input Embedding and Positional Embedding.

Decoder

<img src="E:\Users\DingYi\Desktop\1.27 report\Report_12.28.assets\image-20220126160735124.png" alt="image-20220126160735124" style="zoom:67%;" />

Similar to Encoder, Decoder is also composed of 6 identical layers, and each layer contains 3 parts:

- Multi-Head Self-Attention
- Multi-Head Context-Attention
- Position-Wise Feed-Forward Network

The above three parts have a residual connection (redidual connection), and then a Layer Normalization.

##### Multi-Head Attention

<img src="E:\Users\DingYi\Desktop\1.27 report\Report_12.28.assets\image-20220126160745626.png" alt="image-20220126160745626" style="zoom:50%;" />

In the above figure, Multi-Head Attention is to do the Scaled Dot-Product Attention process H times, and then merge the output.

The formula of the multi-head attention mechanism is as follows:

![[formula]](https://www.zhihu.com/equation?tex=Q_i%3DQW_i%5EQ%2CK_i%3DKW_i%5EK%2CV_i%3DVW_i%5EV%2Ci%3D1%2C...%2C8)

![[formula]](https://www.zhihu.com/equation?tex=head_i%3DAttention%28Q_i%2CK_i%2CV_i%29%2Ci%3D1%2C...%2C8)

![[formula]](https://www.zhihu.com/equation?tex=MultiHead%28Q%2CK%2CV%29%3DConcact%28head_1%2C...%2Chead_8%29W%5EO)

Here, we assume

 ![[formula]](https://www.zhihu.com/equation?tex=Q%2CK%2CV%E2%88%88R%5E%7B512%7D%2CW_i%5EQ%2CW_i%5EK%2CW_i%5EV%E2%88%88R%5E%7B512%5Ctimes64%7D%2CW%5EO%E2%88%88R%5E%7B512%5Ctimes512%7D%2Chead_i%E2%88%88R%5E%7B64%7D)

> ① Input "tinking machine" Sentence
> ② the sentences Tokenize converted Word Embedding X
> ③ The X-cut into 8 parts, and weight and weight ![[formula]](https://www.zhihu.com/equation?tex=W_i)multiplied constitute the input vector ![[formula]](https://www.zhihu.com/equation?tex=W_iX), forming ![[formula]](https://www.zhihu.com/equation?tex=Q_i%2CK_i%2CV_i%2Ci%3D1%2C...%2C8)
> ④ calculated Attention weight matrix ![[formula]](https://www.zhihu.com/equation?tex=z_i%3Dsoftmax%28Q_iK_i%5ET%2F%5Csqrt%7Bd_k%7D%29V_i), and finally each ![[formula]](https://www.zhihu.com/equation?tex=z_i)combined to form ![[formula]](https://www.zhihu.com/equation?tex=Z_i)
> ⑤ Finally, the results of 8 heads ![[formula]](https://www.zhihu.com/equation?tex=Z_i%2Ci%3D1%2C...8)merge ![[formula]](https://www.zhihu.com/equation?tex=Z%5EC%3Dconcact%28Z_1%2C...Z_8%29%2Ci%3D1%2C...%2C8)point by weight ![[formula]](https://www.zhihu.com/equation?tex=W_O), is formed ![[formula]](https://www.zhihu.com/equation?tex=Z%3DZ%5ECW_O)
> as will be seen a lot of the X matrix ![[formula]](https://www.zhihu.com/equation?tex=W_0%5EQ)
>
> <img src="E:\Users\DingYi\Desktop\1.27 report\Report_12.28.assets\image-20220126160802564.png" alt="image-20220126160802564" style="zoom:50%;" />

#### Hybrid deep model for audio-visual emotion recognition

<img src="E:\Users\DingYi\Desktop\1.27 report\Report_12.28.assets\image-20220126161033125.png" alt="image-20220126161033125" style="zoom:50%;" />

##### DBNs:

DBNs are built by stacking multiple Restricted Boltzmann Machines (RBMs) . By using multiple RBMs and the greedy layer wise training algorithm , DBNs can effectively learn a multi-layer generative model of input data. Based on this generative model, the distribution properties of input data can be discovered, and the hierarchical feature representations characterizing input data can be also extracted. Due to such good property, DBNs and its variant called Deep Bolzmann Machines (DBMs)  have been successfully utilized to learn high-level feature representations from low-level hand-crafted features for multimodal emotion recognition.

##### CNNs:

CNNs employ raw image data as inputs instead of handcrafted features. CNNs are mainly composed of convolutional layers and fully connected layers, where convolutional layers learn a discriminative multi-level feature representation from raw inputs and fully connected layers can be regarded as a non-linear classiﬁer. Due to the large-scale available training data and the effective training strategies introduced in recent works, CNNs have exhibited signiﬁcant success in various vision tasks like object detection and recognition

##### Feature Extraction

1.  Audio affective features : prosody features, voice quality features,  spectral features

    ​	Mel-frequency Cepstral Coefﬁcient (MFCC) is the most well-known spectral features

2.  Visual feature extraction methods

    1.  Static : appearance-based feature extraction methods （CNN\LSTM）
    2.  Dynamic : Facial animation parameters or motion parameters

##### Multimodality Fusion

Multimodality fusion is to integrate audio and visual modalities with different statistical properties

1.  Feature-level fusion

    Feature-level fusion is the most common and straightforward way, in which all extracted features are directly concatenated into a single high-dimensional feature vector.

2.  Decision-level fusion

    Decision-level fusion aims to combine several unimodal emotion recognition results through an algebraic combination rule.

3.  Score-level fusion

    Score-level fusion, as a variant of decision-level fusion, has been recently employed for audio-visual emotion recognition

4.  Model-level fusion

    Model-level fusion, as a compromise between feature-level fusion and decision-level fusion, has also been used for audio-visual emotion recognition. This method aims to obtain a joint feature representation of audio and visual modalities. Its implementation mainly depends on the used fusion model.

#### **Video Feature Extraction：**

Two popular techniques for feature extraction are principal component analysis (PCA) and linear discriminant analysis (LDA). PCA is an unsupervised technique and is used to reduce the image dimensionality by providing the core features to represent an image data. LDA is a supervised learning technique and is used to enhance the separation of the extracted features amongst the different classes. PCA and LDA methods have been used extensively in the research area of face recognition. In our approach, the original PCA and LDA algorithms have been further enhanced to improve the recognition efficacy. We used the Bi-directional Principal
Component Analysis [27] and the Least-Square Linear Discriminant Analysis [28] for dimensionality reduction and class discrimination. 





#### algorithms [Title: Emotion recognition using deep learning approach from audio-visual emotional big data]

##### 2D CNN for audio signal

There are four [convolution layers](https://proxy.library.spbu.ru:2068/topics/computer-science/convolution-layer) and three pooling layers. The last layer is a fully-connected [neural network](https://proxy.library.spbu.ru:2068/topics/computer-science/neural-networks) with two hidden layers. 

![1-s2.0-S1566253517307066-gr4_lrg](Report_12.28(1).assets/1-s2.0-S1566253517307066-gr4_lrg.jpg)

In the 2D CNN, there are 64 filters of size 7 × 7 in the first convolution layer, 128 filters of size 7 × 7 in the second convolution layer, and 256 filters of size 3 × 3 in the third convolution layer. The fourth convolution layer has 512 filters of size 3 × 3. The size of filters is chosen to maintain a good balance between phone co-articulatory effect and long vowel phone. The stride in all the cases is 2.

The convolved images are normalized by using an exponential linear unit (ELU) as follows ([Eq. (1)](https://proxy.library.spbu.ru:2068/science/article/pii/S1566253517307066#eqn0001)):

![image-20220127160835260](Report_12.28(1).assets/image-20220127160835260.png)

##### 3D CNN for video signal

For the 3D CNN we have adopted a pre-trained model as described in [[53\]](https://proxy.library.spbu.ru:2068/science/article/pii/S1566253517307066#bib0053). This 3D CNN model was originally developed for sports action recognition purpose. Later, the model was utilized in many video processing applications including emotion recognition from the video [[46\]](https://proxy.library.spbu.ru:2068/science/article/pii/S1566253517307066#bib0046). The structure of the 3D CNN model is shown in [Table 5](https://proxy.library.spbu.ru:2068/science/article/pii/S1566253517307066#tbl0005). There are eight convolution layers and five max-pooling layers. At the end, there are two fully-connected layers, each having 4096 neurons. A softmax layer follows the fully-connected layers. The stride of the filters is one. The input to the model is 16 key frames (RGB) resized to 227 × 227.

#####  ELM-based fusion

The ELM is based on a single hidden layer [feed-forward network](https://proxy.library.spbu.ru:2068/topics/computer-science/feedforward-network) (SHLFN), which was introduced in [[55\]](https://proxy.library.spbu.ru:2068/science/article/pii/S1566253517307066#bib0055). There are some advantages of the ELM over the conventional CNN, such as fast learning, no need for weight adjustment during training, and no overfitting.

![1-s2.0-S1566253517307066-gr5_lrg](Report_12.28(1).assets/1-s2.0-S1566253517307066-gr5_lrg.jpg)

#### **reference**

1. Petridis, S., Stafylakis, T., Ma, P., Tzimiropoulos, G., & Pantic, M. (2018, December). Audio-visual speech recognition with a hybrid ctc/attention architecture. In *2018 IEEE Spoken Language Technology Workshop (SLT)* (pp. 513-520). IEEE.
2. Afouras, T., Chung, J. S., Senior, A., Vinyals, O., & Zisserman, A. (2018). Deep audio-visual speech recognition. *IEEE transactions on pattern analysis and machine intelligence*.
3. S. Watanabe, T. Hori, S. Kim, J. R. Hershey, and T. Hayashi, “Hybrid CTC/attention architecture for end-to-end speech recognition,” IEEE Journal of Selected Topics in Signal Processing, vol. 11, no. 8, pp. 1240–1253, 2017.
4. J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Bengio, “Attention-based models for speech recognition,” in Advances in neural information processing systems, 2015, pp. 577–585.
5. https://zhuanlan.zhihu.com/p/109983672
5. Zhang S, Zhang S, Huang T, et al. Learning affective features with a hybrid deep model for audio–visual emotion recognition[J]. IEEE Transactions on Circuits and Systems for Video Technology, 2017, 28(10): 3030-3043.
5. Hossain M S, Muhammad G. Emotion recognition using deep learning approach from audio–visual emotional big data[J]. Information Fusion, 2019, 49: 69-78.
5. Seng K P, Ang L M, Ooi C S. A combined rule-based & machine learning audio-visual emotion recognition approach[J]. IEEE Transactions on Affective Computing, 2016, 9(1): 3-13.

### The description of XAI 

#### What is XAI?

Nowadays, deep neural networks are widely used in mission critical systems such as healthcare, self-driving vehicles, and military which have direct impact on human lives. However, the black-box nature of deep neural networks challenges its use in mission critical applications, raising ethical and judicial concerns inducing lack of trust. Explainable Artificial Intelligence (XAI) is a field of Artificial Intelligence (AI) that promotes a set of tools, techniques, and algorithms that can generate high-quality interpretable, intuitive, human-understandable explanations of AI decisions.

#### Where is the XAI focusing on?

Scope of explanations can be either local or global. Some methods can be extended to both. Locally explainable methods are designed to express, in general, the individual feature attributions of a single instance of input data x from the data population X. For example, given a text document and a model to understand the sentiment of text, a locally explainable model might generate attribution scores for individual words in the text. Globally explainable models provide insight into the decision of the model as a whole - leading to an understanding about attributions for an array of input data.

#### How is the XAI method developed?

- **Intrinsic:** Explainability is baked into the neural network architecture itself and is generally not transferrable to other architectures.
- **Post-Hoc:** XAI algorithm is not dependent on the model architecture and can be applied to already trained neural networks.

#### Methods and techniques

1. SHapley Additive exPlanations (SHAP)

   A game theoretically optimal solution using Shapley values for model explainability was proposed by Lundberg et al. [64]. SHAP explains predictions of an input x by computing individual feature contributions towards that output prediction. By formulating the data features as players in a coalition game, Shapley values can be computed to learn to distribute the payout fairly.

   In SHAP method, a data feature can be individual categories in tabular data or superpixel groups in images similar to LIME. SHAP then deduce the problem as a set of linear function of functions where the explanation is a linear function of features [89]. If we consider g as the explanation model of an ML model f, z' ∈ {0, 1}^M as the coalition vector, M the maximum coalition size, and φ_j ∈ R the feature attribution for feature j, g(z') is the sum of bias and individual feature contributions such tha

   ![image-20220126162211955](E:\Users\DingYi\Desktop\1.27 report\Report_12.28.assets\image-20220126162211955.png)

   Lundberg et al. [64] further describes several variations to the baseline SHAP method such as KernelSHAP which reduces evaluations required for large inputs on any ML model, LinearSHAP which estimates SHAP values from a linear model’s weight coefﬁcients given independent input features, Low-Order SHAP which is efﬁcient for small maximum coalition size M, and DeepSHAP which adapts DeepLIFT method [59] to leverage the compositional nature of deep neural networks to improve attributions. Since KernelSHAP is applicable to all machine learning algorithms, we describe it in Algorithm 2. The general idea of KernelSHAP is to carry out an additive feature attribution method by randomly sampling coalitions by removing features from the input data and linearizing the model inﬂuence using SHAP kernels.

   **Reference:**

   [64] S. M. Lundberg and S. I. Lee, “A uniﬁed approach to interpreting model predictions,” in Advances in Neural Information Processing Systems, 2017, pp. 4765–4774.

   [59] M. Sundararajan, A. Taly, and Q. Yan, “Axiomatic attribution for deep networks,” in 34th International Conference on Machine Learning, ICML 2017, 2017.

   [89] C. Molnar, Interpretable Machine Learning. Lulu. com, 2020.

2. Local Interpretable Model-Agnostic Explana

   In 2016, Ribeiro et al. introduced Local Interpretable Model-Agnostic Explanations (LIME) [65]. To derive a representation that is understandable by humans, LIME tries to ﬁnd importance of contiguous superpixels (a patch of pixels) in a source image towards the output class. Hence, LIME ﬁnds a binary vector x'∈ {0, 1} to represent the presence or absence of a continuous path or ’superpixel’ that provides the highest representation towards class output. This works on a patch-level on a single data input. Hence, the method falls under local explanations. There is also a global explanation model based on LIME called SP-LIME described in the global explainable model sub section. Here, we focus on local explanations.

   ![image-20220126162238685](E:\Users\DingYi\Desktop\1.27 report\Report_12.28.assets\image-20220126162238685.png)

   Consider g ∈ G, the explanation as a model from a class of potentially interpretable models G. Here, g can be decision trees, linear models, or other models of varying interpretability. Let explanation complexity be measured by Ω(g). If πx(z) is a proximity measure between two instances x and z around x, and L(f, g, πx) represents faithfulness of g in approximating f in locality deﬁned by πx, then, explanation ξ for the input data sample x is given by the LIME equation:

   ![image-20220126162257691](E:\Users\DingYi\Desktop\1.27 report\Report_12.28.assets\image-20220126162257691.png)

   Now, in Equation 7, the goal of LIME optimization is to minimize the locality-aware loss L(f, g, πx) in a model agnostic way. Example visualization of LIME algorithm on a single instance is illustrated in Figure 9. Algorithm 1 shows the steps to explain the model for a single input sample and the overall procedure of LIME. Here, for the input instance we permute data by ﬁnding a superpixel of information (‘fake’ data). Then, we calculate distance (similarity score) between permutations and original observations. Now, we know how different the class scores are for the original input and the new ‘fake’ data.

   **Reference:**

   [65] M. T. Ribeiro, S. Singh, and C. Guestrin, “”Why Should I Trust You?”,” in Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining - KDD ’16. New York, New York, USA: ACM Press, 2016, pp. 1135–1144.

3. Layer-wise Relevance BackPropagation

   LRP technique introduced in 2015 by Bach et al. [69] is used to ﬁnd relevance scores for individual features in the input data by decomposing the output predictions of the DNN. The relevance score for each atomic input is calculated by backpropagating the class scores of an output class node towards the input layer. The propagation follows a strict conservation property whereby a equal redistribution of relevance received by a neuron must be enforced. In CNNs, LRP backpropagates information regarding relevance of output class back to input layer, layer-by-layer. In Recurrent Neural Networks (RNNs), relevance is propagated to hidden states and memory cell. Zero relevance is assigned to gates of the RNN. If we consider a simple neural network with input instance x, a linear output y, and activation output z, the system can be described as:

   ![image-20220126162312794](E:\Users\DingYi\Desktop\1.27 report\Report_12.28.assets\image-20220126162312794.png)

   If we consider R(z_j) as the relevance of activation output, the goal is to get R_i←j, that is to distribute R(z_j) to the corresponding input x:

   ![image-20220126162318044](E:\Users\DingYi\Desktop\1.27 report\Report_12.28.assets\image-20220126162318044.png)

   Final relevance score of individual input x is the summation of all relevance from z_j for input x_i:

   ![image-20220126162323338](E:\Users\DingYi\Desktop\1.27 report\Report_12.28.assets\image-20220126162323338.png)

   **Reference:**

   [69] S. Bach, A. Binder, G. Montavon, F. Klauschen, K.-R. M¨uller, and W. Samek, “On Pixel-Wise Explanations for Non-Linear Classiﬁer Decisions by Layer-Wise Relevance Propagation,” PLOS ONE, vol. 10, no. 7, p. e0130140, Jul 2015.

### Research plan

1. I plan to use the TM-CTC model to train three models,-Audio-Only (AO), Video-Only (VO) and Audio-Visual (AV), on the LRS2 dataset for the audio and visual speech recongition.
2. For the Transformer model, we plan to visualize the weights output in the Multi-Head Attention. Visualize its output Q, K, V. Describe the model with graphics, so as to achieve the interpretation of the model.

