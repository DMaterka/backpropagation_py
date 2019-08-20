<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/latest.js?config=TeX-MML-AM_CHTML"></script>

#  Backpropagation in python

This project is mainly for educational purpose. It will implement neural network backpropagation algorithm.

Currently there are a couple of examples of backpropagation implemented to illustrate few simple problems: 

* bulk computation in neurons - linear algebra, matrix computation
* performance benchmarks
* objective approach 

# Alghorithm

The simple neural network with 3 layers looks like below:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<svg xmlns="http://www.w3.org/2000/svg" style="background-color: rgb(255, 255, 255);" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" width="452px" height="332px" viewBox="-0.5 -0.5 452 332" content="&lt;mxfile modified=&quot;2019-08-20T21:06:10.879Z&quot; host=&quot;www.draw.io&quot; agent=&quot;Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:68.0) Gecko/20100101 Firefox/68.0&quot; etag=&quot;SSCjVyy6I3PVwzKyrKV3&quot; version=&quot;11.2.1&quot; pages=&quot;1&quot;&gt;&lt;diagram id=&quot;pmW9mR4PJnjcFVjbcCM7&quot; name=&quot;Page-1&quot;&gt;7VvbcpswEP0aPyZjJMDwmDhJnZl2kplMJ+kjNQpWKyMXy7d+fUVAXCRoCEkQMXnxaBe0SHsOq5UWj+B0uf8SeavFN+ojMgJjfz+CFyMAJobNf2PFIVFA10kUQYT9RGXkijv8F6XKcardYB+tSzcySgnDq7JyTsMQzVlJ50UR3ZVve6Sk/NSVFyBFcTf3iKq9xz5bJFoHTHL9DOFgIZ5s2G5yZemJm9OZrBeeT3cFFbwcwWlEKUtay/0Ukdh3wi9Jv6uaq9nAIhSyJh1Ogq9Twv58J9s13V6tLiZr9OsktbL1yCadcDpYdhAe4Fa4s7lwvltghu5W3jy+suNwc92CLQmXDN5MTaGIoX3tGI1s5pwxiC4Riw78FtFhkjrrIMm73PdCtSi4Xei8FO0gs5w7hDdSn7zAP6Df/oFAs39gv/wDbYk/hmb/mP32DzA1+8fqt3+gpdk/dr/8Y4Ge8cdp4J/QP4sTAS6FNERln6A9Zg+F9g/eHp9aqXQRT3sshIMQQj7yh6JQ6BWLebcnSfRLRoZ8JeGQvM9HTzfRHD2/LDEvChB7LvyoaBbQsirQEroIEY/hbXm4VRCmT7ilmE8kIwuA0ssksyCZZtqrmLnIhpyaVU8YSvygGHpiVDbt9iRzh0kykW73m2SmFJEsqyXJXMnQWDL0ziQTmUp7lmWMOR0Dt8yaiQmf4c2TdIsizOeBogIjX0pcDQSEWgloWP/nTWMC2mVDhszk9yZgk73OEYa5pmup1acwZxpvFOag3THLmuwYm4a5PKolrLEd+wOGuaYE1BzmJN6YrcOctM66HROwyZb8CMNc08VUb5iTF1O37ZZBWkyz+NkVy5ocbBwhy2BDltk6WQYdp0QOw2kZy0woGYLdxjLw6j3Dx2SZ9SFYJmVaJmwZy0wp94NOt7FMTH1oLDM/BMvks7FxW5bJedmk41imbj+vR8Am3F3n681P3gzipiF0/CEFtUJIhvaszMI1i+hvNKWERjlLHzEhksojOAi5OOeEibcI5/GJOZ575Cy9sMS+T+rO2yO6CX3kl9j4upKfVcbFEDgV+GVW8Es+I327kqi6g6sCCgwOKDlMV9QeuwVKzUFnn29UvHUtA+Xqxkkt8lXhNLgXSsYpqydpA2rSCCg4dKBgRVW4W6DUsvDNZ+TjOad8vFLx+VC3QFWVVhNMfLwVgNxXYHdtzGrhy7QFG8cLqlwd154gQjWTr0FwcEsakL7fA7pzRKgm8zVYDW5VU78s0I2VWrmoxArUR8ajxUr61Eh7Tg/VnL4Gq8HHwKpvmLvFSk3ra7AaXgyUsRJlNW1YqZl9FVYz42ZwMdCU/zvh6H6v1OS+EiswQKwsObfQ/F6Jh718IzaD9fANayMmv4Dw/c6ruJj/cSypwOT/voOX/wA=&lt;/diagram&gt;&lt;/mxfile&gt;"><defs/><g><ellipse cx="65" cy="115" rx="35" ry="35" fill="#ffffff" stroke="#000000" pointer-events="none"/><ellipse cx="65" cy="265" rx="35" ry="35" fill="#ffffff" stroke="#000000" pointer-events="none"/><ellipse cx="255" cy="55" rx="35" ry="35" fill="#ffffff" stroke="#000000" pointer-events="none"/><ellipse cx="255" cy="185" rx="35" ry="35" fill="#ffffff" stroke="#000000" pointer-events="none"/><ellipse cx="255" cy="295" rx="35" ry="35" fill="#ffffff" stroke="#000000" pointer-events="none"/><ellipse cx="415" cy="185" rx="35" ry="35" fill="#ffffff" stroke="#000000" pointer-events="none"/><path d="M 100 265 L 220 185" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="none"/><path d="M 100 115 L 220 185" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="none"/><path d="M 100 115 L 222.03 72.01" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="none"/><path d="M 100 265 L 220 295" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="none"/><path d="M 100 265 L 223.01 68.02" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="none"/><path d="M 100 115 L 220 295" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="none"/><path d="M 290 55 L 380 185" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="none"/><path d="M 290 295 L 380 185" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="none"/><path d="M 290 185 L 380 185" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="none"/><g transform="translate(25.5,70.5)"><switch><foreignObject style="overflow:visible;" pointer-events="all" width="8" height="15" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; font-size: 12px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; vertical-align: top; width: 9px; white-space: nowrap; overflow-wrap: normal; text-align: center;"><div xmlns="http://www.w3.org/1999/xhtml" style="display:inline-block;text-align:inherit;text-decoration:inherit;white-space:normal;">I<sub>1</sub></div></div></foreignObject><text x="4" y="14" fill="#000000" text-anchor="middle" font-size="12px" font-family="Helvetica">[Not supported by viewer]</text></switch></g><g transform="translate(15.5,220.5)"><switch><foreignObject style="overflow:visible;" pointer-events="all" width="8" height="15" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; font-size: 12px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; vertical-align: top; width: 9px; white-space: nowrap; overflow-wrap: normal; text-align: center;"><div xmlns="http://www.w3.org/1999/xhtml" style="display:inline-block;text-align:inherit;text-decoration:inherit;white-space:normal;">I<sub>2</sub></div></div></foreignObject><text x="4" y="14" fill="#000000" text-anchor="middle" font-size="12px" font-family="Helvetica">[Not supported by viewer]</text></switch></g><g transform="translate(212.5,0.5)"><switch><foreignObject style="overflow:visible;" pointer-events="all" width="14" height="15" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; font-size: 12px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; vertical-align: top; width: 15px; white-space: nowrap; overflow-wrap: normal; text-align: center;"><div xmlns="http://www.w3.org/1999/xhtml" style="display:inline-block;text-align:inherit;text-decoration:inherit;white-space:normal;">H<sub>1</sub></div></div></foreignObject><text x="7" y="14" fill="#000000" text-anchor="middle" font-size="12px" font-family="Helvetica">[Not supported by viewer]</text></switch></g><g transform="translate(212.5,140.5)"><switch><foreignObject style="overflow:visible;" pointer-events="all" width="14" height="15" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; font-size: 12px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; vertical-align: top; width: 15px; white-space: nowrap; overflow-wrap: normal; text-align: center;"><div xmlns="http://www.w3.org/1999/xhtml" style="display:inline-block;text-align:inherit;text-decoration:inherit;white-space:normal;">H<sub>2</sub></div></div></foreignObject><text x="7" y="14" fill="#000000" text-anchor="middle" font-size="12px" font-family="Helvetica">[Not supported by viewer]</text></switch></g><g transform="translate(212.5,250.5)"><switch><foreignObject style="overflow:visible;" pointer-events="all" width="14" height="15" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; font-size: 12px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; vertical-align: top; width: 15px; white-space: nowrap; overflow-wrap: normal; text-align: center;"><div xmlns="http://www.w3.org/1999/xhtml" style="display:inline-block;text-align:inherit;text-decoration:inherit;white-space:normal;">H<sub>3</sub></div></div></foreignObject><text x="7" y="14" fill="#000000" text-anchor="middle" font-size="12px" font-family="Helvetica">[Not supported by viewer]</text></switch></g><g transform="translate(387.5,130.5)"><switch><foreignObject style="overflow:visible;" pointer-events="all" width="15" height="15" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; font-size: 12px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; vertical-align: top; width: 16px; white-space: nowrap; overflow-wrap: normal; text-align: center;"><div xmlns="http://www.w3.org/1999/xhtml" style="display:inline-block;text-align:inherit;text-decoration:inherit;white-space:normal;">O<sub>1</sub></div></div></foreignObject><text x="8" y="14" fill="#000000" text-anchor="middle" font-size="12px" font-family="Helvetica">[Not supported by viewer]</text></switch></g><g transform="translate(143.5,70.5)"><switch><foreignObject style="overflow:visible;" pointer-events="all" width="33" height="15" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; font-size: 12px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; vertical-align: top; width: 34px; white-space: nowrap; overflow-wrap: normal; text-align: center;"><div xmlns="http://www.w3.org/1999/xhtml" style="display:inline-block;text-align:inherit;text-decoration:inherit;white-space:normal;"><div>W<sub>I1H1</sub></div></div></div></foreignObject><text x="17" y="14" fill="#000000" text-anchor="middle" font-size="12px" font-family="Helvetica">[Not supported by viewer]</text></switch></g><g transform="translate(133.5,120.5)"><switch><foreignObject style="overflow:visible;" pointer-events="all" width="33" height="15" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; font-size: 12px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; vertical-align: top; width: 34px; white-space: nowrap; overflow-wrap: normal; text-align: center;"><div xmlns="http://www.w3.org/1999/xhtml" style="display:inline-block;text-align:inherit;text-decoration:inherit;white-space:normal;">W<sub>I1H2</sub></div></div></foreignObject><text x="17" y="14" fill="#000000" text-anchor="middle" font-size="12px" font-family="Helvetica">[Not supported by viewer]</text></switch></g><g transform="translate(103.5,160.5)"><switch><foreignObject style="overflow:visible;" pointer-events="all" width="33" height="15" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; font-size: 12px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; vertical-align: top; width: 34px; white-space: nowrap; overflow-wrap: normal; text-align: center;"><div xmlns="http://www.w3.org/1999/xhtml" style="display:inline-block;text-align:inherit;text-decoration:inherit;white-space:normal;">W<sub>I1H3</sub></div></div></foreignObject><text x="17" y="14" fill="#000000" text-anchor="middle" font-size="12px" font-family="Helvetica">[Not supported by viewer]</text></switch></g><g transform="translate(93.5,200.5)"><switch><foreignObject style="overflow:visible;" pointer-events="all" width="33" height="15" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; font-size: 12px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; vertical-align: top; width: 34px; white-space: nowrap; overflow-wrap: normal; text-align: center;"><div xmlns="http://www.w3.org/1999/xhtml" style="display:inline-block;text-align:inherit;text-decoration:inherit;white-space:normal;">W<sub>I2H1</sub></div></div></foreignObject><text x="17" y="14" fill="#000000" text-anchor="middle" font-size="12px" font-family="Helvetica">[Not supported by viewer]</text></switch></g><g transform="translate(133.5,230.5)"><switch><foreignObject style="overflow:visible;" pointer-events="all" width="33" height="15" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; font-size: 12px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; vertical-align: top; width: 34px; white-space: nowrap; overflow-wrap: normal; text-align: center;"><div xmlns="http://www.w3.org/1999/xhtml" style="display:inline-block;text-align:inherit;text-decoration:inherit;white-space:normal;">W<sub>I2H2</sub></div></div></foreignObject><text x="17" y="14" fill="#000000" text-anchor="middle" font-size="12px" font-family="Helvetica">[Not supported by viewer]</text></switch></g><g transform="translate(133.5,275.5)"><switch><foreignObject style="overflow:visible;" pointer-events="all" width="33" height="15" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; font-size: 12px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; vertical-align: top; width: 34px; white-space: nowrap; overflow-wrap: normal; text-align: center;"><div xmlns="http://www.w3.org/1999/xhtml" style="display:inline-block;text-align:inherit;text-decoration:inherit;white-space:normal;">W<sub>I2H3</sub></div></div></foreignObject><text x="17" y="14" fill="#000000" text-anchor="middle" font-size="12px" font-family="Helvetica">[Not supported by viewer]</text></switch></g><g transform="translate(330.5,90.5)"><switch><foreignObject style="overflow:visible;" pointer-events="all" width="38" height="15" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; font-size: 12px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; vertical-align: top; width: 39px; white-space: nowrap; overflow-wrap: normal; text-align: center;"><div xmlns="http://www.w3.org/1999/xhtml" style="display:inline-block;text-align:inherit;text-decoration:inherit;white-space:normal;">W<sub>H1O1</sub></div></div></foreignObject><text x="19" y="14" fill="#000000" text-anchor="middle" font-size="12px" font-family="Helvetica">[Not supported by viewer]</text></switch></g><g transform="translate(310.5,165.5)"><switch><foreignObject style="overflow:visible;" pointer-events="all" width="38" height="15" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; font-size: 12px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; vertical-align: top; width: 39px; white-space: nowrap; overflow-wrap: normal; text-align: center;"><div xmlns="http://www.w3.org/1999/xhtml" style="display:inline-block;text-align:inherit;text-decoration:inherit;white-space:normal;">W<sub>H2O1</sub></div></div></foreignObject><text x="19" y="14" fill="#000000" text-anchor="middle" font-size="12px" font-family="Helvetica">[Not supported by viewer]</text></switch></g><g transform="translate(330.5,240.5)"><switch><foreignObject style="overflow:visible;" pointer-events="all" width="38" height="15" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; font-size: 12px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; vertical-align: top; width: 39px; white-space: nowrap; overflow-wrap: normal; text-align: center;"><div xmlns="http://www.w3.org/1999/xhtml" style="display:inline-block;text-align:inherit;text-decoration:inherit;white-space:normal;"><div>W<sub>H3O1</sub></div></div></div></foreignObject><text x="19" y="14" fill="#000000" text-anchor="middle" font-size="12px" font-family="Helvetica">[Not supported by viewer]</text></switch></g></g></svg>
```
[1] Generaing svg file throught https://www.draw.io/

There are two main steps to compute the next iteration - forward and backward pass. 
Every step has been provided with the data from its related demonstration scripts (examples).

## Forwardpropagation
This step aims to calculate the sum of the neurons and applies activation function to determine if the neuron has been activated. 
Let V(x) be the dot product of previous layer neurons values. 
$$
V(x) = \sum_n I_n \times W_{I_nH_x}
$$

To determine if the neuron activates, now let's apply the activation function. 
The function is well known in neural network world Sigmoid Function 

$$
S(x) = {1 \over 1 + \exp^{-x}}
$$
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

## Author

Daniel Materka