============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given.

Bug reports
===========

When `reporting a bug <https://github.com/bbquercus/deepblink/issues>`_ please include:

    * Your operating system name and version.
    * Any details about your local setup that might be helpful in troubleshooting.
    * Detailed steps to reproduce the bug.

Documentation improvements
==========================

deepblink could always use more documentation, whether as part of the
official deepblink docs, in docstrings, or even on the web in blog posts,
articles, and such. As the build process takes a while, please make sure to
combine multiple changes (such as typos) into a single pull request.

Feature requests and feedback
=============================

The best way to send feedback is to file an issue at https://github.com/bbquercus/deepblink/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that code contributions are welcome :)

Development
===========

To set up `deepblink` for local development:

1. Fork `deepblink <https://github.com/bbquercus/deepblink/>`_
   (look for the "Fork" button).
2. Clone your fork locally::

    git clone git@github.com:BBQuercus/deepBlink.git

3. Create a branch for local development::

    git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.
4. Install the conda development environment::

    conda env create -f environment.yml

   Activate it and work as desired.
5. In case some conda versions are outdated, be sure to update them using: ::

    conda update --all
    pip list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1  | xargs -n1 pip install -U

   Once updated [1]_, export the environment for the next one to use by replacing the current environment.yml with: ::

    conda env export > environment.yml

6. When you're done making changes run all the checks and docs builder with `tox <https://tox.readthedocs.io/en/latest/install.html>`_ one command::

    tox

7. Commit your changes and push your branch to GitHub::

    git add .
    git commit -m "Detailed description; Separated by semicolons; Without trailing period"
    git push origin name-of-your-bugfix-or-feature

8. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

All pull requests are code reviewed by the core development team. For larger changes make sure to:

1. Include passing tests (run ``tox``) [2]_.
2. Update documentation when there's new API, functionality etc.
3. Add a note to ``CHANGELOG.rst`` about the changes.
4. Add yourself to ``AUTHORS.rst``.

.. [1] Thanks to rbp answering question #2720014 on `stackoverflow <https://stackoverflow.com/questions/2720014/how-to-upgrade-all-python-packages-with-pip>`_.
.. [2] If you don't have all the necessary python versions available locally you can rely on Travis - it will
       `run the tests <https://travis-ci.org/github/bbquercus/deepblink/pull_requests>`_ for each change you add in the pull request.

       It will be slower though ...

Tips
----

To run a subset of tests::

    tox -e envname -- pytest -k test_myfeature

To run all the test environments in *parallel* (you need to ``pip install detox``)::

    detox
