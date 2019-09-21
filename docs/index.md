---
title: Constrained Neural Nets Workbook
subtitle: A workbook for investigating constrained neural networks in PyTorch
author: G. Eli Jergensen<br/>Lawrence Berkeley National Labs<br/>Summer 2019
layout: project
---

# Constrained Neural Nets Workbook

As a student researcher at Lawrence Berkeley National Labs during the summer of 2019, I investigated two methods for constraining neural networks with partial differential equations (PDEs). The first method treats neural network training as an optimization problem and converts the training process to a constrained optimization problem. The second method operates on a fully-trained neural network and nonlinearly “projects” the trainable parameters of the network to a valid configuration which satisfies the constraints. Experimentally, we discovered that neither method seemed practically useful, despite promising theoretical guarantees. In light of these negative outcomes, we did not publish our results. However, to aid future researchers investigating this or similar topics, we provide in this repository the code for our project. Additionally, we offer a [writeup][1] of our theory, experimental procedure, and results as well as the [slides][2] for a seminar I gave on the topic. We hope these will provide insight to future experimentation on the topic of constrained neural networks.

{% include image-pair.html url1="/resources/paper.png" description1="Project Writeup" link1="/downloads/Constrained_Neural_Nets_Workbook.pdf" url2="/resources/slides.png" description2="Seminar Slides" link2="/downloads/Neural_Network_Optimization_Under_PDE_Constraints.pdf" class="center-stretch" %}

[1]:{{ site.url }}/{{ site.github.repository_name }}/downloads/Constrained_Neural_Nets_Workbook.pdf
[2]:{{ site.url }}/{{ site.github.repository_name }}/downloads/Neural_Network_Optimization_Under_PDE_Constraints.pdf