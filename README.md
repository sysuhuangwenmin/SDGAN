# SDGAN
An PyTorch implementation of "SDGAN: Disentangling Semantic Manipulation for Facial Attribute Editing" (AAAI-24)

# Abstract
Facial attribute editing has garnered significant attention, yet prevailing methods struggle with achieving precise attribute manipulation while preserving irrelevant details and controlling attribute styles. This challenge primarily arises from the strong correlations between different attributes and the interplay between attributes and identity. In this paper, we propose Semantic Disentangled GAN (SDGAN), a novel method addressing this challenge. SDGAN introduces two key concepts: a semantic disentanglement generator that assigns facial representations to distinct attribute-specific editing modules, enabling the decoupling of the facial attribute editing process, and a semantic mask alignment strategy that confines attribute editing to appropriate regions, thereby avoiding undesired modifications. Leveraging these concepts, SDGAN demonstrates accurate attribute editing and achieves high-quality attribute style manipulation through both latent-guided and reference-guided manners. We extensively evaluate our method on the CelebA-HQ database, providing both qualitative and quantitative analyses. Our results establish that SDGAN significantly outperforms state-of-the-art techniques, showcasing the effectiveness of our approach.

# Introduction
Illustration of correct modification and unrelated preservation (the 1st row) \& style manipulation (the 2nd row) with the proposed SDGAN.
![ES](https://raw.githubusercontent.com/sysuhuangwenmin/SDGAN/main/ES.png)


# Pretrained Model
The model checkpoint can be downloaded using [Google Drive link](https://drive.google.com/file/d/13K3G806OVdyuGi-1eJkjsH3-3b6e4Rm8/view?usp=drive_link). The checkpoint should be located in the path core/outputs/celeba-hq_256/checkpoints.

# Testing

python test.py


# Ackonwledgements
This code refers to the following project:

[1] (https://github.com/imlixinyang/HiSD)
