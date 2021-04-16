Installation
==============

Stable version
-----------------
To install the latest release of ``pulser``, have Python 3.7.0 or higher
installed, then use ``pip``: ::

  pip install pulser


Latest version
---------------
For the latest version of Pulser, you can install Pulser from source by
cloning the `Pulser Github repository <https://github.com/pasqal-io/Pulser>`_,
and entering your freshly created ``Pulser`` directory. There, you'll checkout
the ``develop`` branch - which holds the latest (unstable) version of Pulser -
and install from source by running: ::

  git checkout develop
  pip install -e .

Bear in mind that your installation will track the contents of your local
Pulser repository folder, so if you checkout a different branch (e.g. ``master``),
your installation will change accordingly.

If you want to install the development requirements, follow up by running: ::

  pip install -r requirements.txt
