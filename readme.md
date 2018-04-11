# egtplot: A python package for 3-Strategy Evolutionary Games

This is a software package for plotting and animating three-strategy evolutionary games on a triangular simplex.
* The package can be used to create phase portraits of 3-strategy games
<img src="images/simplex_example.png" width="500" height="400" />

* It can also be used to animate transient behavior of the game  
<img src="images/animation_1.gif" width="400" height="400" />

### Dependencies

This software package depends on the following libraries:

* `numpy>=1.13, scipy, shapely` for computations
* `matplotlib` for static plots
* `imageio, moviepy` for animations
* `tqdm` for progress bar

Installing the package from `pypi` will take care of all the dependencies
```
pip install egtplot
```
### Installation

* The software is available on `pypi`
```
pip install egtplot
```

* You can clone this repository using `git` software
```
git clone https://github.com/mirzaevinom/egtplot.git
```

* Alternatively you can download and extract a zip file of this repo. Then  `egtplot.py` file can be called locally as a module.
```
import egtplot
```

### Usage
* This software has two main functions: `plot_static` for plotting static simplex figures and `plot_animated` for generating simplex animations.
* For detailed usage of these functions we refer to our interactive jupyter notebook: [`egtplot_demonstration.ipynb`](egtplot_demonstration.ipynb)
