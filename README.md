# Cardiac Model

This repository contains the IGA-cardiac-model module called CardIGA (Cardiac IsoGeometric Analysis). It is based on the Isogeometric Analysis paradigm proposed by [Hughes et al. (2005)](https://www.sciencedirect.com/science/article/pii/S0045782504005171). The fundamental idea is to remove the tedious meshing step, typical in traditional finite elements, by higher-order splines and provide to capability to perform analyses directly onto the exact geometry domain. Due to its relatively recent introduction, IGA is not widely used in conventional analysis software. Hence, one has to resort to various (Python, C, C++, ..) modules. CardIGA is build on the [Nutils](https://nutils.org/) Python architecture to provide a user-friendly environment. However, computational optimization is limited as a result.

## Some background

CardIGA is developed by R. Willems during his PhD (2020-2024) and is based on the cardiac-model developed by the Cardio-Vascular Biomechanics (CVBM) group at the Eindhoven University of Technology. The original cardiac model on which CardIGA is based, is build and solved using the [FEniCS](https://fenicsproject.org/) Python library (FEA-based). It has been converted to the [Nutils](https://nutils.org/) Python library which supports IGA (partially meaning that not all functionalities are incorporated yet). CardIGA has been verified using various test-cases, including an in-depth comparison that has been published. The following contributions use CardIGA:

- [An isogeometric analysis framework for ventricular cardiac mechanics. (2024)](https://doi.org/10.1007/s00466-023-02376-x)
- [Echocardiogram-based ventricular isogeometric cardiac analysis using multi-patch fitted NURBS. (2024)](https://doi.org/10.1016/j.cma.2024.116958)
- [Isogeometric-mechanics-driven electrophysiology simulations of ventricular tachycardia. (2023)](https://doi.org/10.1007/978-3-031-35302-4_10)
- [A probabilistic reduced-order modeling framework for patient-specific cardio-mechanical analysis. (2024)](https://doi.org/10.48550/arXiv.2411.08822)


## Workflow

The model relies on a set of NURBS-based geometry files, which define the multi-patch heart. These files are generated by a separate module [vtnurbs](https://gitlab.tue.nl/iga-heart-model/nurbs-geometry) (ventricle template NURBS). The CardIGA model initializes the following using the NURBS geometry as an input:
- Fiber field (analytic or rule-based)
- Passive myocardium properties
- Active myocardium properties
- Hemodynamics (lumped parameter model)

## Structure

The structure of the repository is as follows:
- analyses/    -> Contains useful analyses scripts for left or bi-ventricles, idealized or patinet-specific;
- geometries/  -> Contains a set of example geometries;
- cardiga/     -> Contains the actual IGA cardiac model;
- tests/       -> Various test scripts, primarily for developing purposes;
- utils/       -> Some utility functions. 

## How to use the module

After cloning the repository in your desired local folder, CardIGA can be installed by specifying "pip install -e ." inside the folder where the repository is cloned. The module can then be used by simply importing it: "import cardiga".