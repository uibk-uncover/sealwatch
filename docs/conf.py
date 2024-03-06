# Configuration file for the Sphinx documentation builder.

# -- Project information
project = 'sealwatch'
copyright = '2023, Martin Benes & Benedikt Lorch, University of Innsbruck'
author = 'Martin Benes, Benedikt Lorch'

release = '2024.03'
version = '2023.03.a0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.autosectionlabel',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['templates']

# -- Options for HTML output
html_theme = 'alabaster'
html_static_path = ['static']
html_theme_options = {
    'logo': 'seal.png',
    'logo_name': 'sealwatch',
    'description': 'Modern steganalysis in Python',
    'show_powered_by': False,
}

# -- Options for EPUB output
epub_show_urls = 'footnote'
