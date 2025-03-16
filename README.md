# PCNN with Adaptive Gabor Filter
## _Pulse-coupled Neural Network based on an adaptive Gabor filter for pavement crack segmentation_
### By Antonio Luna-Álvarez and Dante Mújica-Vargas


[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

An adaptive Pulse-coupled Neural Network based on noise estimation was proposed. The model is capable of adapting to the type of image to obtain the best segmentation by filtering textures without prior training.

## Features

- Compared to the original model, it reduces to 2% of the iterations it would take. 
- The algorithm was parallelized on the GPU, reducing the response time in an embedded system from x^n to x. 
- It is capable of processing 10 images in ≈ 0.8 seconds, which makes it feasible to implement in a real-time system. 
- As a result iterations were reduced to 2% with ≈ 90% precision.

## Requirements
- GCC compiler 4.x+
- Cuda Toolkit 10.x+ (NVCC) for parallel processing


## Experimentation
Test image obtained from the [CrackForest database](https://github.com/cuilimeng/CrackForest-dataset).
![](Samples/001.ppm)
Resulting image of the proposed model
![](Results/001.ppm)

## Compilation and execution
The repository offers two alternatives to run the program. The first one is implemented in C sequentially, the second one implements CUDA libraries for parallel processing using GPU. To run the parallel version it is necessary to have the Nvidia hardware and the [CUDA libraries](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html) .

To compile and run the sequential version:

```sh
gcc segmentation.c -lm -std=c99 -o run
./run [image path]
```

To compile and run the parallel version on GPU:

```sh
nvcc segmentation.cu -o run
./run [image path] 
```
Practical example of compilation and execution:
```sh
nvcc segmentation.cu -o run
./run Samples/001.ppm
```

## Citation
We appreciate citing our work in your publications


```
@article{Luna2022pcnn,
  title={Pulse-coupled Neural Network based on an adaptive Gabor filter for pavement crack segmentation},
  author={Luna {\'A}lvarez, Antonio and M{\'u}jica Vargas, Dante and de Jes{\'u}s Rubio, Jos{\'e} and Rosales Silva, Alberto},
  journal={Journal of Applied Research and Technology},
  year={2022}
}
```


## License

MIT

**Free Software**

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [dill]: <https://github.com/joemccann/dillinger>
   [git-repo-url]: <https://github.com/joemccann/dillinger.git>
   [john gruber]: <http://daringfireball.net>
   [df1]: <http://daringfireball.net/projects/markdown/>
   [markdown-it]: <https://github.com/markdown-it/markdown-it>
   [Ace Editor]: <http://ace.ajax.org>
   [node.js]: <http://nodejs.org>
   [Twitter Bootstrap]: <http://twitter.github.com/bootstrap/>
   [jQuery]: <http://jquery.com>
   [@tjholowaychuk]: <http://twitter.com/tjholowaychuk>
   [express]: <http://expressjs.com>
   [AngularJS]: <http://angularjs.org>
   [Gulp]: <http://gulpjs.com>

   [PlDb]: <https://github.com/joemccann/dillinger/tree/master/plugins/dropbox/README.md>
   [PlGh]: <https://github.com/joemccann/dillinger/tree/master/plugins/github/README.md>
   [PlGd]: <https://github.com/joemccann/dillinger/tree/master/plugins/googledrive/README.md>
   [PlOd]: <https://github.com/joemccann/dillinger/tree/master/plugins/onedrive/README.md>
   [PlMe]: <https://github.com/joemccann/dillinger/tree/master/plugins/medium/README.md>
   [PlGa]: <https://github.com/RahulHP/dillinger/blob/master/plugins/googleanalytics/README.md>
