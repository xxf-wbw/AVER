#### What was already done in literature?

This paper presents a facial expression recognition framework using LEMHI-CNN and CNN-RNN. The integrated framework incorporates facial landmarks to enable attention-aware facial motion capturing and utilize [neural networks](https://proxy.library.spbu.ru:2068/topics/computer-science/neural-networks) to extract spatial-temporal features and classify them, which achieves better performance than most of the state-of-the-art methods on CK+, MMI and AFEW dataset. Our main contributions are threefold. Firstly, we proposed an attention-aware facial motion features based on MHI. Secondly, we introduced temporal segment LSTM to video emotion recognition and improve it. Thirdly, we integrated two models with late fusion based on random weight search.

#### What was possible to do?

How to improve the performance on wild expression dataset, such as AFEW, needs to be further explored.

### Algorithms

1. #### Cross temporal segment LSTM

   /home/alex/桌面/AVER/report23/1.jpg

2. Integrated framework of LEMHI-CNN and CNN-RNN

   /home/alex/桌面/AVER/report23/2.jpg

   

   Ubiquitous Emotion

   Data 

   video BP4D+ multimodal data

   The Ryerson Audio-Visual Database of Emotional Speech and Song

   Pre-provessing

   Video:

   ​	Use Haar features to detect the face and scale it 256*256

   Audio:

   ​	plot the raw audio signal onto the 2D image plane and scale it 256*256

   Algorithms:

   CNN convolutional neural networks

   Inception V3 CNN with 3 convolutional layers of size 32 64 and 128

   Already done :

   presented a method for recognizing emotion using audio and video data, including a method for representing raw audio signals as a plotted waveform.

   First, is to use the raw audio signals by splitting them into blocks of time and using this raw data to train our deep networks. Second, is the fusion of the modalities. This can be done by creating a new image from the face and audio images. This approach to image fusion has shown success in face recognition

   Improve :

   Our approach to multimodal fusion of audio and video data can address this by providing a strong multimodal representation of emotion.

Multiple Spatio-temporal Feature Learning for Video-based Emotion Recognition in the Wild