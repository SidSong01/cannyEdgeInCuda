## Building

To build the project, there is a makefile supplied. Simply typing `make` would compile the program and create an executable.

## Running the Program

Run it as:

`./cannyEdgeDetc \<inputimage\> \<output\>`


## Possible Configurations
The function prepareKernelCall() contains the calls to each of the four implementation of
the edge detection algorithm. Comment or uncomment the desired function, and also uncomment/comment
the necessary preprocessor directives described in the function.

## Results

[//]: # (Image References)

[image1]: ./test01.jpg
[image2]: ./out.jpg
[image3]: ./result.png

### Input

![alt text][image1]

### Output

![alt text][image2]

### Run

![alt text][image3]
