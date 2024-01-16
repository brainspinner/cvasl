==================================
Medical Professional Documentation
==================================

This is the present documentation home **for medical professionals** who do not spend most of their time coding. Please visit the [Tutorial](https://exploreasl.github.io/Documentation) for a detailed overview of the steps required for processing.

Introduction
============

Accelerated biological aging is associated with cognitive decline and neurodegenerative disease and can be identified through neuroimaging methods and machine learning. This brain-age estimation method, _CVASL_, currently incorporates traditional structural MRI markers of accelerated biological aging with cerebrovascular health markers, a key player in the evolution of cognitive dysfunction. Cerebrovascular markers are extracted from Arterial Spin Labelling MRI ASL-MRI data, which was shown to improve the brain-age accuracy of structural-alone data ([Dijsselhof et al.,](https://onlinelibrary.wiley.com/doi/10.1002/hbm.26242)).

The _CVASL_ package requires input features based on structural and ASL MRI measurements along with age and sex information to produce estimations of MRI-derived biological brain age. The input to _CVASL_ is ideally the output of the [ExploreASL package](https://exploreasl.github.io/Documentation). However, processing by any other software is possible as long as the formatting requirements are kept.

The package utilizes 'training' data (healthy individuals with a large age range) and 'testing' data (to assess biological brain age in). The user can choose from a selection of machine learning algorithms, data harmonization methods, and visualizations. It is further possible to pool from different MRI datasets. 

Additional functions of the package allow for visualisation of MRI images, input image features, and results of the brain age predictions.
