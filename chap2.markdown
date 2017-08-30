1. Is it okay to initialize all the weights to the same value as long as that value is selected randomly using He initialization?
> `NO`
all weights should be initialized independently. They should not
all have the same initial value. One important goal of sampling weights
randomly is to break symmetries: if all the weights have the same initial
value, all the neurons in any given layer will always have the same weights.
It's just like having one neuron per layer.

2. Is it okay to initialize the bias terms to 0?
> Yes

3. Name three advantages of the ELU activation function over ReLU.

>
1. Will not Die.
2. Smooth everywhere.
3. `take on negative values.`

4. In which cases would you want to use each of the following activation functions: ELU, leaky ReLU (and its variants), ReLU, tanh, logistic, and softmax?

>
+ ELU: computation speed is quickly, such as have plenty GPUs; or, Training
size is not very huge, which could not led to a time assuming computation.
+ Leaky Relu: Need quickly computation, and gradient will die in some cases.
There I will use Leaky Relu.
+ Relu: Training need speed up or Training number is very huge.
+ tanh: if you need to output a number between -1 and 1, but nowadays it is
not used much in hidden layers.
+ logistic: classify one-zero targets, or for probability.
+ softmax: classification multiply targets.

