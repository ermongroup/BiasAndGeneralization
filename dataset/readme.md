# Dots Dataset
This dataset contains images of 1-12 colored dots. Each object count contains 65k example images. 
The dataset can be downloaded at
[google](https://drive.google.com/open?id=1CsDMOIGEsD1l3BLhuQDfEfEmLEb83wMz) 
[dropbox](https://www.dropbox.com/s/4qhnpxzct3fkvxh/dots.zip?dl=0).

![dots_example](img/dots_example.png)

For example usage, download the dataset, and launch example.ipynb. 
Replace '<dataset_path>' with the actual parent directory of the extracted dataset (this directory should contain folders '1_dots', '2_dots', ...).

# Pie Dataset

This dataset generates a pie shape with different colors. Example images are shown below. 
This dataset is generated on-the-fly and do not require pre-download or pre-processing. 
For example usage, see example.ipynb. 

![pie_example](img/pie_example.png)

The parameters for the dataset is encoded as a sequence of 5 integers 'abcde', where

- a: the number of different colors. Takes values in 1-4.
- b: the size of the pie. Takes values in 1-9, larger value indicates larger size.
- c: the x translation of the pie. Takes values in 1-9.
- d: the y translation of the pie. Takes values in 1-9.
- e: the proportion of red colors. Takes values in 1-9, larger values indicate higher proportion of red color.
        
In addition a,b,c,d can take value 0. When the value is zero, it is independently uniformly sampled from [1, 9] for each image.
