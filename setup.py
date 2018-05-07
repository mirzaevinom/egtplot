from distutils.core import setup
setup(
  name = 'egtplot',
  packages = ['egtplot'], # this must be the same as the name above
  version = '0.4.2',
  description = 'a package for plotting 3-strategy evolutionary game solutions',
  author = 'Inom Mirzaev and Drew Williamson',
  author_email = 'mirzaev@colorado.edu',
  url = 'https://github.com/mirzaevinom/egtplot', # use the URL to the github repo
  download_url = 'https://github.com/mirzaevinom/egtplot/archive/0.4.2.tar.gz', # I'll explain this in a second
  keywords = ['egt', 'simplex', 'evolutionary game theory', 'evolution'], # arbitrary keywords
  classifiers = [],
  install_requires=['matplotlib', 'scipy', 'shapely', 'imageio', 'moviepy', 'tqdm'],
  include_package_data=True
)