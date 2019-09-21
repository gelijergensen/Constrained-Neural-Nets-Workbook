---
title: Constrained Neural Nets Workbook
subtitle: A workbook for investigating constrained neural networks in PyTorch
author: G. Eli Jergensen<br/>Lawrence Berkeley National Labs<br/>Summer 2019
layout: project
---

# Constrained Neural Nets Workbook

As a student researcher at Lawrence Berkeley National Labs during the summer of 2019, I investigated two methods for constraining neural networks with partial differential equations (PDEs). The first method treats neural network training as an optimization problem and converts the training process to a constrained optimization problem. The second method operates on a fully-trained neural network and nonlinearly “projects” the trainable parameters of the network to a valid configuration which satisfies the constraints. Experimentally, we discovered that neither method seemed practically useful, despite promising theoretical guarantees. In light of these negative outcomes, we did not publish our results. However, to aid future researchers investigating this or similar topics, we provide in this repository the code for our project. Additionally, we offer a [writeup][1] of our theory, experimental procedure, and results as well as the [slides][2] for a seminar I gave on the topic. We hope these will provide insight to future experimentation on the topic of constrained neural networks.


{% include image-pair.html url1="/resources/paper.png" description1="Project Writeup" link1="/downloads/Constrained_Neural_Nets_Workbook.pdf" url2="/resources/slides.png" description2="Seminar Slides" link2="/downloads/Neural_Network_Optimization_Under_PDE_Constraints.pdf" class="center-stretch" %}
 
<!-- {% include image.html url="resources/example-plot.png" description="A caption for the figure" class="center-stretch" %} -->


<!-- # Neural network optimization under PDE constraints

We examine two methods of applying multiple equality constraints to neural networks influenced by dynamical
systems and differential geometry. The first method can be applied directly to constrain neural network
training and is shown to be equivalent to a particular choice of Lagrange multipliers, enabling use with
standard backpropagation techniques. We evaluate the speed of this method in light of the known theoretical
guarantees and propose a second method which trades guarantees for speed. The second method for
constraining neural networks can be applied to a model post-training and is therefore also completely
independent of the model design and architecture. Experimentally, we evaluate the performance and
computational efficiency of these methods against both unconstrained and soft-constrained baselines on a
simple toy problem which allows for detailed investigation. We primarily investigate the Helmholtz equation
as a linear partial differential equation (PDE) constraint, as many constraints for scientific domains can be
framed as PDEs. We show that while the outputs of the constrained models do sometimes seem qualitatively
better and are less prone than soft-constraints to over-constraining the problem, all methods seem to be
unpromising in practice, despite theoretical guarantees. Finally, we discuss difficulties of implementing these
methods for practical problems and offer suggestions for future improvements. -->

<!-- 
Something vaguely abstract-y will go here, _e.g._

We review a method of applying multiple hard equality constraints to neural network training influenced by dynamical systems and differential geometry. We slightly modify the method to better allow for use with neural networks. Specifically, we show that the algorithm is equivalent to a particular Lagrange multiplier system, which allows for the construction of a modified loss function and for the application of standard backpropagation. We evaluate the speed of the algorithm in light of the known theoretical guarantees and propose further modifications which trades off some of the guarantees for a several orders of magnitude increase in speed. Experimentally, we evaluate the performance and computational efficiency of the original algorithm and its modification against both unconstrained and “soft-constrained” baselines on a simple toy problem which allows for detailed investigation of the convergence properties. Further, we examine multiple constraints in the form of linear and non-linear partial differential equations (PDEs), such as the linear Helmholtz equation and a non-linear differential form of the Pythagorean equation. We show that while the outputs of the constrained models do often seem qualitatively better and are not as prone as the soft-constraint method to collapsing to a small subspace of the valid constraint space, the theoretical guarantees of convergence do not seem to be practically evident. Lastly, we discuss the implication of the computational complexity on practical application and offer suggestions for the implementation and possible future improvements.


# Constrained Optimization of Neural Networks

Here is where the description of Experiment A will go. Here is an example of using MathJax:

$$ mean = \frac{\displaystyle\sum_{i=1}^{n} x_{i}}{n} $$

# Projection of Neural Networks to Constraint Manifolds

Here is where the description of Experiment B will go. Here is an example plot:

{% include image.html url="resources/example-plot.png" description="A caption for the figure" class="center" %}

And here's the same image, but stretched:

{% include image.html url="resources/example-plot.png" description="A caption for the figure" class="center-stretch" %}

This is a link to [download the PDF][1] -->

[1]:/downloads/Constrained_Neural_Nets_Workbook.pdf
[2]:/downloads/Neural_Network_Optimization_Under_PDE_Constraints.pdf