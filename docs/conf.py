# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, under development
#
import sys, subprocess
import os


sys.path.insert(0, os.path.abspath('..'))
# -- Project information -----------------------------------------------------

project = 'cvasl'
copyright = '2023, c.moore@esciencecenter.nl'
author = 'c.moore@esciencecenter.nl'

# The full version, including alpha/beta/rc tags
try:
    tag = subprocess.check_output([
        'git',
        '--no-pager',
        'describe',
        '--abbrev=0',
        '--tags',
    ]).strip().decode()
except subprocess.CalledProcessError as e:
    print(e.output)
    tag = 'v0.0.0'

release = tag[1:]


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.extlinks',
    'sphinx.ext.imgmath',
    'sphinx.ext.intersphinx',

]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'agogo'

html_theme_options = {
    #"bgcolor": Background color.
    #"bodyfont":t (CSS font family): Font for normal text.
    "headerfont": "Helvetica",
    "rightsidebar": "False", 
    "headerbg": "#8B0000", 
    "footerbg": "800000",
    "linkcolor": "#8B0000", 
    "headercolor1": "#8B0000", 
    "headercolor2": "#8B0000", 
    "headerlinkcolor":"black", 
    
    # "sidebarbtncolor" :"#8B0000", #Background color for the sidebar collapse button (used when collapsiblesidebar is True).
    # "sidebartextcolor (CSS color): Text color for the sidebar.
    # "sidebarlinkcolor (CSS color): Link color for the sidebar.
    # "relbarbgcolor (CSS color): Background color for the relation bar.
    # "relbartextcolor (CSS color): Text color for the relation bar.
    # "relbarlinkcolor (CSS color): Link color for the relation bar.
    # "bgcolor (CSS color): Body background color.
    # "textcolor (CSS color): Body text color.
    # "linkcolor (CSS color): Body link color.
    # "visitedlinkcolor (CSS color): Body color for visited links.
    # "headbgcolor (CSS color): Background color for headings.
    # "headtextcolor (CSS color): Text color for headings.
    # "headlinkcolor (CSS color): Link color for headings.
    # "codebgcolor (CSS color): Background color for code blocks.
    # "codetextcolor (CSS color): Default text color for code blocks, if not set differently by the highlighting style.
    # "bodyfont (CSS font-family): Font for normal text.
    # "headfont (CSS font-family): Font for headings.
}

# Add any paths that contain custom static files (such as style sheets) here, #
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


intersphinx_mapping = {
    'python': (
        'https://docs.python.org/{.major}'.format(
            sys.version_info,
        ),
        None,
    ),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'matplotlib': ('http://matplotlib.org', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'IPython': ('https://ipython.readthedocs.io/en/stable/', None),
}

