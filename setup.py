from distutils.core import setup
setup(
  name = 'egtplot',
  packages = ['egtplot'],
  version = '0.4.7',
  description = 'a package for plotting 3-strategy evolutionary game solutions',
  author = 'Inom Mirzaev and Drew Williamson',
  author_email = 'mirzaev@colorado.edu',
  url = 'https://github.com/mirzaevinom/egtplot',
  download_url = 'https://github.com/mirzaevinom/egtplot/archive/0.4.7.tar.gz',
  keywords = ['egt', 'simplex', 'evolutionary game theory', 'evolution'],
  classifiers = [],
  install_requires=['matplotlib', 'scipy', 'shapely', 'imageio', 'moviepy', 'tqdm', 'requests'],
  include_package_data=True
)