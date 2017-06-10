# DeepSpectroscopy
Spectroscopy with Deep Learning

Analyzing spectrum with deep learning.

In recent years, deep learning has attracted an incresing attention in a wide range of research areas. However, as far as we know, there was no application of deep learning in the field of spectroscopy. So I would like to give an example to demonstrate the usage of deep learning to do qualitative and quantitative analysis of spectral data.

We use Laser Induced Breakdown Spectroscopy (LIBS) to illustrate the whole procecedure.

The spectroscopy files are organized as:

data/1
      /1
      /2
      /3
      ...
    /2
      /1
      /2
      /3
      ...
    /3
      /1
      /2
      /3
      ...
    /4
      /1
      /2
      /3
      ...
      
There are four classes of samples in all which are corresponding to the subfolder 1, 2, 3 and 4.
