### Main research questions

1. Do operator adapted derivative features improve the accuracy of simulations compared to an RBF representation?
2. Does preconditioning help when solving non-linear problems?

It turns out that derivative representers improve the accuracy of approximations by an order of magnitude. 

![alt text](image.png)

![alt text](image-1.png)

Furthermore, preconditiong with a whitenting transform and the Gamblets transform are ridiculously more efficient than optimization in the canonical bases. 

#### Canonical bases

![alt text](without_representers_non_linear_poisson.png)

#### Gamblets bases

![alt text](without_representers_non_linear_poisson_Gamblets.png)

#### Square root or whitening bases

![alt text](without_representers_non_linear_poisson_whitening.png)
