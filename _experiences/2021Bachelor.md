---
title: "Two Person Emotion Interaction Detection and Generation on Video Sequences"
collection: 'experiences'
permalink: /experiences/2021Bachelor
date: "Feb. 2021 - Jun. 2021"
venue: 'SPEIT, Shanghai Jiao Tong, China'
---
In the process of human-robot interaction, how to generate robots’ actions response corresponding to interaction scenarios has become a widespread problem. Predecessors' work mainly focused on the recognition and classification of emotional actions, which seldom got involved in the state of emotion generation. In fact, the continuous expression of human emotions in high-dimensional space makes the emotion generated in the action dimension particularly complicated. Our paper combines emotion prediction with human body pose estimation, uses the component emotion model as a criterion for emotion quantification and constructs an emotional coordinate system of affiliation-dominance for estimating the emotional state of two-person interaction scene; then, based on the interactive sequential skeleton data and emotional tags, we establish sequential models and generative-adversarial models in order to predict a single-person action sequence adapted to the interactive state. We devote to apply this method to the emotion generation of humanoid robots so that to improve the human experience in the process of human-computer interaction. Sequential model and adversarial model that we built are: LSTM-single frame model, GRU-nine frame model and conditional GAN-five frame model. The experiment is conducted on the BoLD dataset containing twenty-six emotional labels and the SBU dataset containing eight interactive actions. The results show that when the dataset is small, sequential model have better performance than adversarial model.

## Network Architecture
![](../images/2021Bachelor/LSTM.png)
> The architecture of LSTM network

![](../images/2021Bachelor/GRU.png)
> The architecture of GRU network

![](../images/2021Bachelor/GAN.png)
> The architecture of GAN

