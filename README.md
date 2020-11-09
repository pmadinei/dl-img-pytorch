# Introduction
During this project a deep neural network has been designed and implemented for products' images classification by utilizing PyTorch built-in functions. Additionally, all the hyperparameters in a neural network, including learning rate, L2 regularization method, activation function, normalizing data, weight initialization, batch size, epoch size, and momentum have been tuned for better results.

## Dataset
You can download the dataset from [HERE](https://drive.google.com/file/d/1aFjcSk9hBzsHusrjO3UjCg4xQo9Ubdr1/view?usp=sharing)

Here is a sample from dataset with labels:

![Sample](https://github.com/pmadinei/dl-img-pytorch/blob/master/docs/Samples.png)

# Requirement
* Python > 3.7
* Jupyter Notebook
* PyTorch 1.6.0
* Scikit-Learn 0.23.2
* Numpy 1.19
* Pandas 1.1.2
* Matplotlib 3.3.2

# Implementation & Result

I used ZIP.id to download dataset to my notebook, and then unzip the main folder. I transformed images into what is called gray-Scaled images. The "imshow" function, displays images in given labels and classes, "make_unique_images" makes uniques cattegories so that I can display each product without repeating ourselves, and lastly, I executed these functions. The "plot_distribution" function takes the data_loader and classes, and uses "draw_plot" to plot the amount of images from each class. 

You can find all reports and codes in jupyter notebook in eithor [HERE](https://github.com/pmadinei/dl-img-pytorch/blob/master/PyTorch%20NN%20Image%20Classification.ipynb) as ipynb 

or [HERE](https://github.com/pmadinei/dl-img-pytorch/blob/master/docs/Report.html) as HTML in your local computer. 

Feel free to contact me if you have any questions about the present project. GOOD LUCK!
