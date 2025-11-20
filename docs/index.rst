Scikit-downscale: toolkit for statistical downscaling
=====================================================

Scikit-downscale is a toolkit for statistical downscaling using Scikit-Learn_.
It is meant to support the development of new and existing downscaling
methods in a common framework. It implements Scikit-learn's `fit`/`predict` API
facilitating the development of a wide range of statistical downscaling models.
Utilities and a high-level API built on Xarray_ and Dask_ support both
point-wise and global downscaling applications.

.. _Xarray: http://xarray.pydata.org
.. _Scikit-Learn: https://scikit-learn.org
.. _Dask: https://dask.org

Quick Start
-----------

Install scikit-downscale:

.. code-block:: bash

   pip install scikit-downscale

Then try your first downscaling example:

.. code-block:: python

   from skdownscale.pointwise_models import QuantileMapper

   # Initialize the model
   qm = QuantileMapper()

   # Fit on training data
   qm.fit(model_data, observations)

   # Generate downscaled predictions
   downscaled = qm.predict(model_data)

Ready to learn more? Check out our :doc:`tutorials/getting-started`!

Documentation Structure
-----------------------

This documentation is organized following the `Diátaxis framework <https://diataxis.fr/>`_:

📚 **Tutorials** - *Learning-oriented*
   Step-by-step lessons to learn scikit-downscale. Start here if you're new!

🔧 **How-to Guides** - *Problem-oriented*
   Practical guides for accomplishing specific tasks.

📖 **Background** - *Understanding-oriented*
   Explanations of concepts, theory, and design decisions.

📋 **Reference** - *Information-oriented*
   Complete API documentation and technical specifications.

.. toctree::
   :maxdepth: 1
   :caption: Documentation

   tutorials/index
   how-to/index
   background/index
   api

.. toctree::
   :maxdepth: 1
   :caption: Project Info

   roadmap

Under Active Development
~~~~~~~~~~~~~~~~~~~~~~~~

Scikit-downscale is under active development. We are looking for additional
contributors to help fill out the list of downscaling methods supported here.
We are also looking to find collaborators interested in using deep learning
to build global downscaling tools. Get in touch with us on our
`GitHub page <https://github.com/pangeo-data/scikit-downscale>`_.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
