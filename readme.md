# egtplot: A python package for 3-Strategy Evolutionary Games
As a very general model of cooperation and competition, evolutionary game theory (EGT) is well-suited to quantitative investigations of the dynamics of interactions between populations. EGT has been used to model phenomena from disparate areas of study, from [poker](http://www.mdpi.com/2073-4336/7/4/39) to [hawks and doves](https://www.nature.com/articles/246015a0) to [cancer](https://www.nature.com/articles/bjc2011517). In many biological EGT models, payoffs are taken to represent the resources a particular organism can extract from its environment given its interaction in that environment with another organism utilizing the same or perhaps a different strategy.

This is a software package for plotting and animating three-strategy evolutionary games.

<img src="images/simplex_example.png" width="440" height="400" /> <img src="images/animation_1.gif" width="400" height="400" />

### Dependencies

This software package depends on the following libraries:

* `numpy>=1.13, scipy, shapely` for computations
* `matplotlib` for static plots
* `imageio, moviepy` for animations
* `tqdm` for progress bar

### Installation

* The software is also available on `pypi`
```
pip install egtplot
```

* You can clone this repository using `git` software
```
git clone
```

* Alternatively you can copy `egtplot.py` file locally and start using as a module.
```
import egtplot
```

### Usage
* This software has two main functions: `plot_static` for plottting static simplex figures and `plot_animated` for generating simplex animations.
* For detailed usage of these functions we refer to our interactive jupyter notebook: [`demonstration.ipynb`](demonstration.ipynb)
