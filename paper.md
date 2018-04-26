---
title: 'egtplot: A Python Package for Three-Strategy Evolutionary Games'
tags:
- python
- evolutionary game theory
- visualization
authors:
- name: Inom Mirzaev
  orcid: 0000-0003-1493-1802
  affiliation: "1, 2"
- name: Drew FK Williamson
  orcid: 0000-0003-1745-8846
  affiliation: 2
- name: Jacob G Scott
  orcid: 0000-0003-2971-7673
  affiliation: "2, 3"
affiliations:
- name: Mathematical Biosciences Institute, The Ohio State University
  index: 1
- name: Department of Translational Hematology and Oncology Research, Cleveland Clinic Foundation
  index: 2
- name: Department of Radiation Oncology, Cleveland Clinic Foundation
  index: 3
date: 25 April 2018
bibliography: joss_bib.bib
---

# Summary

In the study of evolutionary game theory (EGT), there exists a need for open source software for the visualization of the dynamics of games. The Python package `egtplot` represents an attempt to make simple, visual analysis of a particular class of evolutionary games--those with three players and which can be represented by a payoff matrix. Using `egtplot`, these games can be visualized as static images or as animations with a variety of parameters to control what information is presented in the graphic.

EGT is a reformulation of classical game theory wherein the players of the game are members of a population. These members do not choose a strategy, but instead are born with their strategy ingrained, i.e., they cannot change strategy during the game. In biological terms, the strategies might represent discrete species or genotypes, hawks and doves being a classic example. Payoffs the players gain or lose based on their interactions with other players increase or decrease their fitness, thereby influencing the number or proportion of members playing that strategy in the next generation. As such, the populations of strategies can wax and wane as they outcompete or are outcompeted by other strategies.

As a very general model of cooperation and competition, EGT is well-suited for quantitative investigations of the dynamics of interactions between populations. EGT has been used to model phenomena from disparate areas of study, from poker [@javarone2016modeling] to hawks and doves [@smith1973logic] to  host-parasite coevolution [@schenk2017chaotic].

# Example Usage in Research

While there are many use cases for this software, our group's particular focus is on the mathematical modeling of cancer and cancer therapies through evolutionary game theory [@kaznatcheev2017cancer, @kaznatcheev2015edge, @Kaznatcheev179259]. For details on analytical treatments of evolutionary games, please see Artem Kaznatcheev's blog at [Theory, Evolution, and Games Group](https://egtheory.wordpress.com/).

We demonstrate the features of `egtplot` via an example drawn from work modelling the interactions between cancer and healthy cells. Consider the following game from [@basanta2012investigating] which describes the evolutionary game between three strategies of cells, labeled "S" (Stroma), "D" (micronenvironmentally dependent), and "I" (microenvironmentally independent):

<center>

|  |  S | D | I |
| ---- | :----: | :----: | :----: |
| **S** | 0 | $\alpha$ | 0 |
| **D** | $1 + \alpha - \beta$ | $1 - 2 \beta$ | $1 - \beta + \rho$ |
| **I** | $1 - \gamma$ | $1 - \gamma$ | $1 - \gamma$ |

</center>

where $\alpha$ is the benefit derived from the cooperation between a S cell and a D cell, $\gamma$ is the cost of being microenvironmentally independent, $\beta$ is the cost of extracting resources from the microenvironment, and $\rho$ is the benefit derived by a D cell from paracrine growth factors produced by I cells. This paper studies how the healthy cells that make up the majority of the prostate can cooperate and compete with mutant protate cells to produce a clinically-detectable prostate cancer.

Using our package, we can quickly and easily analyze this game numerically and visually. To start, let us choose some simple values for each parameter: $\alpha = 1$, $\beta = 1$, $\gamma = 1$, and $\rho = 1$. __Figure 1__ illustrates the output of the static visualization with the default parameters. This simplex depicts stable equilibria within the S-D and D-I edges, unstable equilibria at each vertex, and shows that every initial condition on the S-I edge is a stable equilibria.

\begin{figure}[h!]\centering
  {\includegraphics[width=\textwidth]{images/output_8_1.png}}
  \caption{Standard output of the package for $\alpha = 1$, $\beta = 1$, $\gamma = 1$, and $\rho = 1$.}
\end{figure}

# Altering Plot Outputs

By altering the default values for the plotting function, a variety of different plotting styles can be achieved as demonstrated in __Figure 2__.

\begin{figure}[h!]\centering
  {\includegraphics[width=\textwidth]{images/two_parts.png}}
  \caption{\textit{Left} Background of the simplex is colored by the speed at which the points would travel along their trajectories. \textit{Right} Displaying the paths taken by each initial condition.}
\end{figure}

Additionally, multiple parameter values can be easily combined into subplots of a larger image. In __Figure 3__, we vary $\alpha$ and $\beta$ values to see how they would independently affect the dynamics of the game.

\begin{figure}[h!]\centering
  {\includegraphics[width=\textwidth]{images/output_20_1.png}}
  \caption{Parameter sweep for different $\alpha$ and $\beta$ values.}
\end{figure}

Finally, `egtplot` also has functionality to display animated versions of these plots. For further information on the use of the software and how these plots were created, please see the [documentation](https://github.com/mirzaevinom/egtplot) and [example notebook](https://github.com/mirzaevinom/egtplot/blob/master/egtplot_demonstration.ipynb).

# Acknowledgements

The authors would like to thank Mathematical Biosciences Institue (MBI) at Ohio State University, for partially supporting this research. MBI receives its funding through the National Science Foundation grant DMS 1440386. We gratefully acknowledge the work of Hanna Schenk whose code on her [GitHub](https://github.com/HannaSchenk/RQchaos) inspired this project.

# Authorship

Inom Mirzaev and Drew FK Williamson contributed equally to this work.

# References
