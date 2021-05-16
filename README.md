# Assignment 2 - Back Propagation

> Submitted by Naman Shrimali

## Target
Rewrite the whole excel sheet showing backpropagation. Explain each major step, and write it on Github. 
* Use exactly the same values for all variables as used in the class
* Take a screenshot, and show that screenshot in the readme file
* Excel file must be there for us to cross-check the image shown on readme (no image = no score)
* Explain each major step
* Show what happens to the error graph when you change the learning rate from [0.1, 0.2, 0.5, 0.8, 1.0, 2.0] 

## Mini Neural Network
![Mini Neural Network](assets/mini_model_diagram.png)
This is our mini model !

### Key Features
It's a mini neural network, which
* Takes 2 inputs
* Has a total of 8 weights
* Does not have any biases 
* Uses ReLU as the activation function
* Uses Gradiant descent algorithm for updating weights
* Uses _L2_ Loss

## Results
The excel sheet uploaded in the repository contains in-depth calculations for learning rates 0.1, 0.2, 0.5, 0.8, 1.0, 2.0.
<br>

![Total loss vs Iterations](assets/lr_loss_vs_iterations.png)
Total loss vs iterations
<br><br>

![Back Propagation Calculation](assets/calc_back_prop.png)
Calculation for Back Propagation


## Explaination
Before we can start with training of our mini model, it is important to get the forward propagation calculations sorted out. Then we can jump to back propagation !
```
i1, i2              : Inputs
o1, o2              : Outputs
h1, h2              : Neurons
a_h1, a_h2          : Activated Neurons (Neurons passed from activation function)
a_o1, a_o2          : Activated Outputs (Outputs passed from activation function)
w1, w2, .... , w8   : Weights

======= Calculations for forward Propagation =======
h1 = i1 * w1 + i2 * w2		
h2 = i1 * w3 + i2 * w4		
a_h1 = σ(h1)		
a_h2 = σ(h2)		
o1 = a_h1 * w5 + a_h2* w6		
o2 = a_h1 * w7 + a_h2* w8		
a_o1 = σ(o1) => 1/(1+exp(-o1))		
a_o2 = σ(o2)		
E1 = ½ * (t1 - a_o1)2		
E2 = ½ * (t2 - a_o2)2		
E_TOTAL = E1 + E2		
======= End of forward propagation calculation =======

Now that we've sorted our our calculations for forward propagation, let's jump to updation of weights (This is where things will get interesting !)

In order to calculate updated weights, it's important to know the concept of partial derivatives


```	


## What's next

I'm done :D

