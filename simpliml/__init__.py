from .about import get_about

__date__ = get_about()['__date__']
__author__ = get_about()['__author__']
__version__ = get_about()['__version__']
__email__ = get_about()['__email__']
__description__ = get_about()['__description__']
__keywords__ = get_about()['__keywords__']
__url__ = get_about()['__url__']
__license__ = get_about()['__license__']
__status__ = get_about()['__status__']
del get_about
