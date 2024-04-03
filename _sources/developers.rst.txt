=================
Developer's Guide
=================

Sending Your Work
=================

We accept pull requests made through GitHub. As is usual,
we request that the changes be rebased
on the branch they are to be integrated into (usually the develop brannch).
We also request that you pre-lint and test anything you send.

We'll try our best to attribute
your work to you, however, you need to release your work under
compatible license for us to be able to use it.

.. warning::

   We don't use `git-merge` command, and if your submission has merge
   commits, we'll have to remove them.  This means that in such case
   commit hashes will be different from those in your original
   submission.


Setting Up Development Environment
==================================


The Way We Do It
^^^^^^^^^^^^^^^^

If you want to develop using Anaconda Python, you would:

Follow the instructions on the readme.

This will create a virtual environment, build `conda` package, install
it and then add development dependencies to what was installed.



The Traditional Ways
^^^^^^^^^^^^^^^^^^^^

Regardless of the downsides of this approach, we try to support more
common ways to work with Python projects.  It's a common practice to
"install" a project during development by either using `pip install
--editable` command, or by using `conda` environment files.

We provide limited support for approaches not based on Anaconda right
now.  For instance, if you want to work on the project using `pip`,
you could try it, and contact us: brainspinner@gmail.com

Due to the complexity of environments involved we reccomend 
using `mamba` a drop in substitute for conda.

The environment files are generated using:


.. code-block:: bash

   conda env create -f ./environment.yml


or
.. code-block:: bash

   mamba env create -f ./environment.yml



Rationale
^^^^^^^^^

There are several problems with the traditional way Python programmers are
taught to organize their development environment.  The way a typical
Python project is developed, it is designed to support a single
version of Python, rarely multiple Python distributions or operating
systems. We are working to support multiple Pythons. Pending. But for
now we are doing what is simple and fast.



Testing
=======

You may now locally run:

.. code-block:: bash

  pytest



Under the hood, this runs unittest and pytest tests.

Additionally, you can automatically test
all notebooks by switching into 
the mriland environment (or any appropriate
environment with nbmake) and running the following command line:

.. code-block:: bash

   pytest --nbmake <<directory_of_notebooks_you_want_to_test>>



Style Guide for Python Code
^^^^^^^^^^^^^^^^^^^^^^^^^^^

We have linting!

.. code-block:: bash

   python ./setup.py lint



Continuous Integration
^^^^^^^^^^^^^^^^^^^^^^

This project has CI setup that uses GitHub Actions
platform.  


.. _GitHub repo: https://github.com/brainspinner/cvasl
.. _GitHub Actions dashboard: https://github.com/brainspinner/cvasl/actions


Style
^^^^^

When it comes to style, beyond linting we are trying
to conform, more or less, to the Google Python style
https://google.github.io/styleguide/pyguide.html


Vendor
^^^^^^

We have a module with variuos submodules called vendor.
This is a place for certain open source code from third party libraries.
Due to the lack of version control and releases on much
scientific software we release code from these libraries with our own.
We include the original licenses, and code that has been 
modified to fit the standard style of Python or whichever
language the original code was written in. These are not complete
versions of the libraries, rather relevant functions.
