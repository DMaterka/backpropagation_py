#  Backpropagation in python

This project is mainly for educational purpose. It will implement neural network backpropagation algorithm.

Currently there are a couple of examples of backpropagation implemented to illustrate few simple problems: 

* bulk computation in neurons - linear algebra, matrix computation
* performance benchmarks
* objective approach 

# Alghorithm

The simple neural network with 3 layers looks like below:

![diagram](./assets/diagram1.png)

There are two main steps to compute the next iteration - forward and backward pass. 
Every step has been provided with the data from its related demonstration scripts (examples).

## Forwardpropagation
This step aims to calculate the sum of the neurons and applies activation function to determine if the neuron has been activated. 
Let V(x) be the dot product of previous layer neurons values. 

![equation1](./assets/eq1.png)

To determine if the neuron activates, now let's apply the activation function. 
The function is well known in neural network world Sigmoid Function 

![equation2](./assets/eq2.png)

[2] todo source

hence the Activated value of the neuron used in further computations is equal to $$ A(x) = S(V(x)) $$

## Backpropagation
@TODO

## Example1

Simple backpropagation network problem of grouping of xor outputs with non-linear separation function.

This demonstrates the weighed neurons in action. There is a sigmoid function used and the inputs are only 1,1

```
python3 example1.py
```

The data comes from the page: 

[3] http://stevenmiller888.github.io/mind-how-to-build-a-neural-network/

The computation from [3] and in Example1.py script of this program are cross validated so all the weights and sums are 
identical except of input layer, 
which has been changed from 1 to 0.99 and from 0 to 0.01 in order to avoid division over zero issues. 

## Features

* Demonstrate backpropagation algorithm in action
* Store the outputs in well structured database
* Use for solving simple problems
* Educational purpose of the program
* Learning python.

## License

This project is released under the MIT Licence.

## References

1. http://latex2png.com/ generating equation images from latex

2. https://www.draw.io/  crating and generating diagram to/from svg

## Author

Daniel Materka