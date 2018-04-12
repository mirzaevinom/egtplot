# egtplot: A Python Package for 3-Strategy Evolutionary Games

This is a software package for plotting and animating three-strategy evolutionary games on a triangular simplex.
* The package can be used to create phase portraits of 3-strategy games

<img src="images/simplex_example.png" width="500" height="400" />

* It can also be used to animate transient behavior of the game  

<img src="images/animation_1.gif" width="400" height="400" />

### Dependencies

The program is written and tested on Python >=3.5. This software package depends on the following libraries:

* `numpy>=1.13, scipy, shapely` for computations
* `matplotlib` for static plots
* `imageio, moviepy` for animations
* `tqdm` for progress bar

Installing the package from `PyPi` will take care of all the dependencies
```
pip install egtplot
```
### Installation

* Easiest way to install the package is through `PyPi`
```
pip install egtplot
```

* You can clone this repository using `git` software and run setup.
```
git clone https://github.com/mirzaevinom/egtplot.git
cd egtplot
python setup.py install
```

* Alternatively you can download and extract a zip file of this repo:
```
cd egtplot-master
python setup.py install
```

### Usage
* This software has two main functions: `plot_static` for plotting static simplex figures and `plot_animated` for generating simplex animations.
* For detailed usage of these functions we refer to our interactive jupyter notebook: [`egtplot_demonstration.ipynb`](egtplot_demonstration.ipynb)
* We also welcome comments and questions regarding our whitepaper on [bioRxiv](https://www.biorxiv.org/content/early/2018/04/12/300004) which describes the package and its usage. The content is nearly identical to the jupyter notebook linked above.

### Citation
If you use this program to do research that leads to publication, we ask that you acknowledge use of this program by citing the following in your publication:

```
Mirzaev I., Williamson D. and Scott J., egtplot: A Python Package for 3-Strategy Evolutionary Games, https://doi.org/10.1101/300004
```

### Acknowledgements

* This material is based upon work supported by the National Science Foundation under Agreement No. 0931642 (Mathematical Biosciences Institute at Ohio State University).
* We gratefully acknowledge the work of Hanna Schenk whose code on her [github](https://github.com/HannaSchenk/RQchaos) inspired this project.
