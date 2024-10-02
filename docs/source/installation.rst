Installation
==============

.. warning::
  Pulser v0.6 introduced a split of the ``pulser`` package that prevents
  it from being correctly upgraded. If you have an older version of ``pulser`` installed
  and wish to upgrade, make sure to uninstall it first by running: ::

    pip uninstall pulser

  before proceeding to any of the steps below.

To install the latest release of ``pulser``, have Python 3.9 or higher
installed, then use ``pip``: ::

  pip install pulser

The standard ``pulser`` distribution will install the core ``pulser`` package
and the ``pulser_simulation`` extension package, which is required if you want
to access the :doc:`apidoc/simulation` features.

If you wish to install only the core ``pulser`` features, you can instead run: ::

  pip install pulser-core


Development version
--------------------
For the development version of Pulser, you can install Pulser from source by
cloning the `Pulser Github repository <https://github.com/pasqal-io/Pulser>`_,
and entering your freshly created ``Pulser`` directory. There, you'll checkout
the ``develop`` branch - which holds the development (unstable) version of Pulser -
and install from source by running: ::

  git checkout develop
  make dev-install

Bear in mind that your installation will track the contents of your local
Pulser repository folder, so if you checkout a different branch (e.g. ``master``),
your installation will change accordingly.

If you want to install the development requirements, stay inside the same ``Pulser``
directory and follow up by running: ::

  pip install -r dev_requirements.txt
