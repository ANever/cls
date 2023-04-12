# Import the version and print the license unless verbosity is disabled
from .version import __version__, __versiondate__, __license__

# Import the actual model
from .basis     import * 
from .solution  import *
