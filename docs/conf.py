# Configuration file for the Sphinx documentation builder.

# -- Project information
project = 'sealwatch'
copyright = '2025–2026, Martin Benes & Benedikt Lorch, University of Innsbruck'
author = 'Martin Benes, Benedikt Lorch'

release = '2026.06'
version = '2026.06'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.autosectionlabel',
    'sphinx_copybutton',
    'sphinx_design',
]
autodoc_typehints = 'description'
# class docstring and constructor docstring
autoclass_content = 'both'
# not the docstrings of torch, e.g., of nn.Module.__init__
autodoc_inherit_docstrings = False

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://docs.pytorch.org/docs/stable/', None),
}
intersphinx_disabled_domains = ['std']

# section labels unique across pages, e.g., guide/loading:Installation
autosectionlabel_prefix_document = True

templates_path = ['templates']

# -- Options for HTML output
html_theme = 'pydata_sphinx_theme'
html_static_path = ['static']
html_css_files = ['css/custom.css']
html_title = 'sealwatch'
html_logo = 'static/seal.png'
html_favicon = 'static/seal.png'
html_theme_options = {
    'logo': {
        'text': 'sealwatch',
    },
    'github_url': 'https://github.com/uibk-uncover/sealwatch',
    'icon_links': [
        {
            'name': 'PyPI',
            'url': 'https://pypi.org/project/sealwatch/',
            'icon': 'fa-brands fa-python',
        },
    ],
    'navbar_align': 'left',
    'show_toc_level': 2,
    'navigation_with_keys': False,
    'footer_start': ['copyright'],
    'footer_end': [],
}
html_show_sourcelink = False

# no section navigation on pages without subpages
html_sidebars = {
    'faq': [],
    'glossary': [],
}

# copy code without the >>> and ... prompts
copybutton_prompt_text = r'>>> |\.\.\. |\$ '
copybutton_prompt_is_regexp = True

# -- Options for EPUB output
epub_show_urls = 'footnote'


# -- Module docstrings

def _strip_authors(app, what, name, obj, options, lines):
    """Drops Author and Affiliation lines from module docstrings."""
    if what == 'module':
        lines[:] = [l for l in lines if not l.startswith(('Author:', 'Affiliation:'))]


def _numpy_alias(app, env, node, contnode):
    """Resolves np.* in types, e.g., np.ndarray, as numpy.* of intersphinx."""
    target = node.get('reftarget', '')
    if target.startswith('np.'):
        from sphinx.ext import intersphinx
        node['reftarget'] = 'numpy.' + target[len('np.'):]
        return intersphinx.missing_reference(app, env, node, contnode)


def setup(app):
    app.connect('autodoc-process-docstring', _strip_authors)
    app.connect('missing-reference', _numpy_alias)
