# Introduction

Examples of affect-sensitive multimodal HCI（human-computer interaction) systems include the following:

1. the system of Lisetti and Nasoz [85], which combines facial expression and physiological signals to recognize the user's emotions, like fear and anger, and then to adapt an animated interface agent to mirror the user's emotion,
2. the multimodal system of Duric et al. [39], which applies a model of embodied cognition that can be seen as a detailed mapping between the user's affective states and the types of interface adaptations,
3. the proactive HCI tool of Maat and Pantic [89], which is capable of learning and analyzing the user's context-dependent behavioral patterns from multisensory data and of adapting the interaction accordingly,
4. the automated Learning Companion of Kapoor et al. [72], which combines information from cameras, a sensing chair, and mouse, wireless skin sensor, and task state to detect frustration in order to predict when the user needs help, and
5. the multimodal computer-aided learning system1 in the Beckman Institute, University of Illinois, Urbana-Champaign (UIUC), where the computer avatar offers an appropriate tutoring strategy based on the information of the user's facial expression, keywords, eye movement, and task state.

# Human Affect (Emotion) Perception

In summary, a large number of studies in psychology and linguistics confirm the correlation between some affective displays (especially prototypical emotions) and specific audio and visual signals (e.g., [1], [47], and [113]). The human judgment agreement is typically higher for facial expression modality than for vocal expression modality. However, the amount of the agreement drops considerably when the stimuli are spontaneously displayed expressions of affective behavior rather than posed exaggerated displays. In addition, facial expression and the vocal expression of emotion are often studied separately. This precludes finding evidence of the temporal correlation between them. On the other hand, a growing body of research in cognitive sciences argues that the dynamics of human behavior are crucial for its interpretation (e.g., [47], [113], [116], and [117]). For example, it has been shown that temporal dynamics of facial behavior represent a critical factor for distinction between spontaneous and posed facial behavior (e.g., [28], [47], [135], and [134]) and for categorization of complex behaviors like pain, shame, and amusement (e.g., [47], [144], [4], and [87]). Based on these findings, we may expect that the temporal dynamics of each modality (facial and vocal) and the temporal correlations between the two modalities play an important role in the interpretation of human naturalistic audiovisual affective behavior. However, these are virtually unexplored areas of research.

Another largely unexplored area of research is that of context dependency. The interpretation of human behavioral signals is context dependent. For example, a smile can be a display of politeness, irony, joy, or greeting. To interpret a behavioral signal, it is important to know the context in which this signal has been displayed, i.e., where the expresser is (e.g., inside, on the street, or in the car), what the expresser's current task is, who the receiver is, and who the expresser is [113].

# The state of the art

## Database

Audio and/or Visual Databases of Human Affective Behavior

1. affect elicitation method (i.e., whether the elicited affective displays are posed or spontaneous),
2. size (the number of subjects and available data samples),
3. modality (audio and/or visual),
4. affect description (category or dimension),
5. labeling scheme, and
6. public accessibility.

![4468714-table-1-source-large](A%20Survey%20of%20Affect%20Recognition%20Methods%20Audio%20Visual%20and%20Spontaneous%20Expressions.assets/4468714-table-1-source-large-1643832263916-1643832283435.gif)

The Cohn-Kanade facial expression database [71] is the most widely used database for facial expression recognition. The BU-3DFE database of Yin and colleagues [148] contains 3D range data of six prototypical facial expressions displayed at four different levels of intensity. The FABO database of Gunes and Piccardi [63] contains videos of facial expressions and body gestures portraying posed displays of basic and nonbasic affective states (six prototypical emotions, uncertainty, anxiety, boredom, and neutral). The MMI facial expression database [106], [98] is, to our knowledge, the most comprehensive data set of facial behavior recordings to date.

## Vision-Based Affect Recognition

![image-20220202231335616](A%20Survey%20of%20Affect%20Recognition%20Methods%20Audio%20Visual%20and%20Spontaneous%20Expressions.assets/image-20220202231335616-1643832832382.png)

This table provides an overview of the currently existing exemplar systems for vision-based affect recognition with respect to the utilized facial features, classifier, and performance. While summarizing the performance of the surveyed systems, we also mention a number of relevant aspects, including the following:

1. type of the utilized data (spontaneous or posed, the number of different subjects, and sample size),
2. whether the system is person dependent or independent,
3. whether it performs in a real-time condition,
4. what the number of target classification categories is,
5. whether and which other cues, aside from the face, have been used in the classification (head, body, eye, posture, task state, and other contexts),
6. whether the system processes still images or videos, and
7. how accurately it performs the target classification.

The current state of the art in the field is listed as follows:

- Methods have been proposed to detect attitudinal and nonbasic affective states such as confusion, boredom, agreement, fatigue, frustration, and pain from facial expressions (e.g., [69], [72], [129], [147], and [87]).
- Initial efforts were conducted to analyze and automatically discern posed (deliberate) facial displays from genuine (spontaneous) displays (e.g., [135] and [134]).
- First attempts are reported toward the vision-based analysis of spontaneous human behavior based on 3D face models (e.g., [123] and [149]), based on fusing the information from facial expressions and head gestures (e.g., [27] and [134]), and based on fusing the information from facial expressions and body gestures (e.g., [61]).
- Few attempts have also been made toward the context-dependent interpretation of the observed facial behavior (e.g., [50], [69], [72], and [104]).
- Advanced techniques in feature extraction and classification have been applied and extended in this field. A few real-time robust systems have been built (e.g., [11]) thanks to the advance of relevant techniques such as real-time face detection and object tracking.

## Audio-Based Affect Recognition

![image-20220202231850140](A%20Survey%20of%20Affect%20Recognition%20Methods%20Audio%20Visual%20and%20Spontaneous%20Expressions.assets/image-20220202231850140-1643833153568.png)

This table provides an overview of the currently existing exemplar systems for audio-based affect recognition with respect to the utilized auditory features, classifier, and performance.

The current state of the art in the research field of automatic audio-based affect recognition can be summarized as follows:

- Methods have been proposed to detect nonbasic affective states, including coarse affective states such as negative and nonnegative states (e.g., [83]), application-dependent affective states (e.g., [3], [12], [65], [79], [157], and [125]), and nonlinguistic vocalizations like laughter and cry (e.g., [133], [91], and [97]).
- A few efforts have been made to integrate paralinguistic features and linguistic features such as lexical, dialogic, and discourse features (e.g., [12], [35], [57], [83], [86], and [120]).
- Few investigations have been conducted to make use of contextual information to improve the affect recognition performance (e.g., [52] and [86]).
- Few reported studies have analyzed the affective states across languages (e.g., [94] and [133]).
- Some studies have investigated the influence of ambiguity of human labeling on recognition performance (e.g., [3] and [86]) and proposed measures of comparing human labelers and machine classifiers (e.g., [125]).
- Advanced techniques in feature extraction, classification, and natural language processing have been applied and extended in this field. Some studies have been tested on commercial call data (e.g., [83] and [35]).

## Audiovisual Affect Recognition

![image-20220202232810124](A%20Survey%20of%20Affect%20Recognition%20Methods%20Audio%20Visual%20and%20Spontaneous%20Expressions.assets/image-20220202232810124.png)

This table provides an overview of the currently existing exemplar systems for audiovisual affect recognition with respect to the utilized auditory and visual features, classifier, and performance.

In summary, research on audiovisual affect recognition has witnessed significant progress in the last few years as follows:

- Efforts have been reported to detect and interpret nonbasic genuine (spontaneous) affective displays in terms of coarse affective states such as positive and negative affective states (e.g., [151]), quadrants in the evaluation-activation space (e.g., [17], [53], and [74]), and application-dependent states (e.g., [122], [154], [97], and [108]).
- Few studies have been reported on efforts to integrate other affective cues aside from the face and the prosody such as body and lexical features (e.g., [53] and [74]).
- Few attempts have been made to recognize affective displays in specific naturalistic settings (e.g., in a car [66]) and in multiple languages (e.g., [138]).
- Various multimodal data fusion methods have been investigated. In particular, some advanced data fusion methods have been proposed, such as HMM-based fusion (e.g., [124], [154], and [150]), NN-based fusion (e.g., [53] and [74]), and BN-based fusion (e.g., [122]).

## 