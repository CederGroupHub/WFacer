.. _contributing :

====================================
Contributing Guidelines
====================================

For optimal teamwork, it's crucial to set clear and pragmatic guidelines upfront,
rather than addressing the confusion that arises from overlooking them later.
If you're committed to making impactful contributions, please take a moment to
thoroughly review the following guidelines!

Bugs, issues, input, questions, etc
===================================
Please use the
`issue tracker <https://github.com/CederGroupHub/WFacer/issues>`_ to share any
of the following:

-   Bugs
-   Issues
-   Questions
-   Feature requests
-   Ideas
-   Input

Having these reported and saved in the issue tracker is very helpful to make
them properly addressed. Please be as descriptive and neat as possible when
opening up an issue. When available, please also attach your I/O data and the
full error message.

Developing
==========
Code contributions can be anything from fixing the simplest bugs, to adding new
extensive features or subpackages. If you have written code or want to start
writing new code that you think will improve **WFacer** then please follow the
steps below to make a contribution.

Guidelines
----------

* All code should have unit tests.
* Code should be well documented following `google style <https://google.github.io/styleguide/pyguide.html>`_  docstrings.
* All code should pass the pre-commit hook. The code follows the `black code style <https://black.readthedocs.io/en/stable/>`_.
* Additional dependencies should only be added when they are critical or if they are
  already a :mod:`smol` or a :mod:`sparse-lm` dependency. More often than not it is best to avoid adding
  a new dependency by simply delegating to directly using the external packages rather
  than adding them to the source code.
* Implementing new features should be more fun than tedious.

Installing a development version
--------------------------------

#. *Clone* the main repository or *fork* it and *clone* clone your fork using git.
   If you plan to contribute back to the project, then you should create a fork and
   clone that::

        git clone https://github.com/<USER>/WFacer.git

   Where ``<USER>`` is your github username, or if you are cloning the main repository
   directly then ``<USER> = CederGroupHub``.

#. Install Python 3.8 or higher. We recommend using python 3.9 or higher:
   `conda <https://docs.conda.io/en/latest/>`_.

#. We recommend developing using a virtual environment. You can do so using
   `conda <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_
   or using `virtualenv <https://docs.python.org/3/tutorial/venv.html>`_.

#. Install the development version of *WFacer* in *editable* mode::

    pip install --verbose --editable .[dev,test]

   This will install the package in *editable* mode, meaning that any changes
   you make to the source code will be reflected in the installed package.

Adding code contributions
-------------------------

#.  If you are contributing for the first time:

    * Install a development version of *WFacer* in *editable* mode as described above.
    * Make sure to also add the *upstream* repository as a remote::

        git remote add upstream https://github.com/CederGroupHub/WFacer.git

    * You should always keep your ``main`` branch or any feature branch up to date
      with the upstream repository ``main`` branch. Be good about doing *fast forward*
      merges of the upstream ``main`` into your fork branches while developing.

#.  In order to have changes available without having to re-install the package:

    * Install the package in *editable* mode::

         pip install -e .


#.  To develop your contributions you are free to do so in your *main* branch or any feature
    branch in your fork.

    * We recommend to only your forks *main* branch for short/easy fixes and additions.
    * For more complex features, try to use a feature branch with a descriptive name.
    * For very complex features feel free to open up a PR even before your contribution is finished with
      [WIP] in its name, and optionally mark it as a *draft*.

#.  While developing we recommend you use the pre-commit hook that is setup to ensure that your
    code will satisfy all lint, documentation and black requirements. To do so install pre-commit, and run
    in your clones top directory::

        pre-commit install

    *  All code should use `google style <https://google.github.io/styleguide/pyguide.html>`_ docstrings
       and `black <https://black.readthedocs.io/en/stable/?badge=stable>`_ style formatting.

#.  Make sure to test your contribution and write unit tests for any new features. All tests should go in the
    ``WFacer\tests`` directory. The CI will run tests upon opening a PR, but running them locally will help find
    problems before::

        pytests tests


#.  To submit a contribution open a *pull request* to the upstream repository. If your contribution changes
    the API (adds new features, edits or removes existing features). Please add a description to the
    `change log <https://github.com/CederGroupHub/WFacer/blob/main/CHANGES.md>`_.

#.  If your contribution includes novel published (or to be published) methodology, you should also edit the
    citing page accordingly.


Adding examples
---------------

In many occasions novel use of the package does not necessarily require introducing new source code, but rather
using the existing functionality, and possibly external packages (that are are requirements) for particular or
advanced calculations.

#.  Create a sub-directory with a descriptive name in the ``docs/src/example_scripts`` directory.
#.  Implement the functionality with enough sections to carefully describe the background, theory,
    and steps in the index.rst file.
#.  Once the script is ready, add an entry to the :ref:`examples` page's rst file so your example shows up in the
    documentation.
