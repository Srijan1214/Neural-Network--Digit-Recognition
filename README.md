# Introduction
This is a neural network I wrote to recognize the `MNIST` handwritten digits dataset. This was my first attempt in making a neural network and I primarily made this program to aide in my understanding of how all of this works.

I did not use any matrix libraries as I felt that for a beginner they provided an extra layer of complexity in understanding the fundamentals. I felt that thinking about each and every neuron would make learning better.

I did not copy any other similar projects. As far as I know, how I implemented the neural network is pretty unique. I used actual graphs and traversed through those graphs, instead of representing the weights and biases performing the operations.

However, this approach did have an affect on the performance of my neural network. I did not care much about this because the project was primary made with the goal of learning.

# How To Compile And Run
If on Linux or Mac, use the Makefile in the "Neural Network starter" folder.
If on Windows, use the make.bat script or load the project into Visual Studio.

Basically, just compile all the `.cpp` and `.h` files into one executable binary.

# Checking The Output Files
I have made it so that the program will test the neural network every epoch of training with the `MNIST` testing data. This test will be outputted into the "Epoch_ Testing" files, where the "_" is the file number. These files are interesting and you can see how the neural network performs.
# Program Output
![Output Image](Output.png)
# Acknowledgment
I loved Michael Nielsen's excellent book "Neural Networks and Deep Learning" with the information of everything in how neural networks work. I built this software by learning from that book. link: http://neuralnetworksanddeeplearning.com/

I got the dataset from here:
http://yann.lecun.com/exdb/mnist/