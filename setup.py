from setuptools import setup
import os, re

long_description = """
CardIGA is an Isogeometric Analysis (IGA) tool/module designed for cardiac mechanics. It relies on the Nutils
package to construct the individual components of the multiphysics problem and solve it monolithically. The
input geometry should be multipatch (nutils) with specified boundary names. Input geometries can be generated
by the vtnurbs package (idealized ventricle geometries) and the ebdm package (error-based-deformation-method)
for patient-specific geometries.
"""
with open(os.path.join('cardiga', '__init__.py')) as f:
  version = next(filter(None, map(re.compile("^version = '([a-zA-Z0-9.]+)'$").match, f))).group(1)

setup(name     = 'cardiga',
      version  = '1.0',
      author   = 'Robin Willems',
      packages = ['cardiga'],
      description      = 'Isogeometric analysis (IGA) for cardiac mechanics',
      download_url     = 'https://gitlab.tue.nl/iga-heart-model/cardiac-model/-/tree/main',
      long_description = long_description,
      zip_safe = False)