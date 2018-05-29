FROM jupyter/base-notebook

USER root

# Install all OS dependencies for fully functional notebook server
RUN apt-get update && apt-get install -yq --no-install-recommends \
    build-essential \
    emacs \
    git \
    inkscape \
    jed \
    libsm6 \
    libxext-dev \
    libxrender1 \
    lmodern \
    netcat \
    pandoc \
    python-dev \
    texlive-fonts-extra \
    texlive-fonts-recommended \
    texlive-generic-recommended \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-xetex \
    unzip \
    nano \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Switch back to jovyan to avoid accidental container runs as root
USER $NB_UID

# install egtplot through pip to get dependencies
RUN pip install egtplot

# clone the egtplot repository to get the notebook, etc.
RUN git clone https://github.com/mirzaevinom/egtplot.git

# install ffmpeg through imageio
RUN python -c "import imageio.plugins.ffmpeg as ff; ff.download()"