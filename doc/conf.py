# Configuration file for the Sphinx documentation  adapted from TorchSpatiotemporal project (https://github.com/TorchSpatiotemporal/tsl/blob/main/docs/source/conf.py).

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
import datetime
import doctest
import os
import sys

from docutils import nodes

sys.path.insert(0, os.path.abspath('../'))
import torch_concepts as pyc

# -- Project information -----------------------------------------------------

project = 'pytorch_concepts'
author = 'PyC Team'
copyright = f'{datetime.datetime.now().year}, {author}'

version = pyc.__version__
release = pyc.__version__


# -- General configuration ---------------------------------------------------

master_doc = 'index'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_design',
    'sphinxext.opengraph',
    'sphinx_copybutton',
    'myst_nb',
    'hoverxref.extension',
]

autosummary_generate = True
autosummary_imported_members = True

source_suffix = '.rst'
master_doc = 'index'

templates_path = ['_templates']

doctest_default_flags = doctest.NORMALIZE_WHITESPACE
autodoc_member_order = 'bysource'

rst_context = {'pyc': pyc}

add_module_names = False
# autodoc_inherit_docstrings = False

# exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

napoleon_custom_sections = [("Shape", "params_style"),
                            ("Shapes", "params_style")]

numfig = True  # Enumerate figures and tables

# Ensure proper navigation tree building
html_show_sourcelink = True
html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/scroll-start.html",
        "sidebar/navigation.html",
        "sidebar/ethical-ads.html",
        "sidebar/scroll-end.html",
    ]
}

# -- Options for intersphinx -------------------------------------------------
#

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pd': ('https://pandas.pydata.org/docs/', None),
    'PyTorch': ('https://pytorch.org/docs/stable/', None),
    'pytorch_lightning': ('https://lightning.ai/docs/pytorch/latest/', None),
    'PyG': ('https://pytorch-geometric.readthedocs.io/en/latest/', None)
}

# -- Theme options -----------------------------------------------------------
#

html_title = "Torch Concepts"
html_theme = 'furo'
language = "en"

html_baseurl = ''
html_static_path = ['_static']
html_logo = '_static/img/logos/pyc.png'
html_favicon = '_static/img/logos/pyc.svg'

html_css_files = [
    'css/custom.css',
]

html_js_files = [
    'js/theme-logo-switcher.js',
]

html_theme_options = {
    "sidebar_hide_name": True,
    "navigation_with_keys": True,
    "collapse_navigation": False,
    "top_of_page_button": "edit",
    "light_css_variables": {
        "color-brand-primary": "#20b0d6",
        "color-brand-content": "#20b0d6",
    },
    "dark_css_variables": {
        "color-brand-primary": "#20b0d6",
        "color-brand-content": "#20b0d6",
        "color-background-primary": "#020d1e",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/pyc-team/pytorch_concepts",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
}

pygments_style = "tango"
pygments_dark_style = "material"

# -- Notebooks options -------------------------------------------------------
#

nb_execution_mode = 'off'
myst_enable_extensions = ['dollarmath']
myst_dmath_allow_space = True
myst_dmath_double_inline = True
nb_code_prompt_hide = 'Hide code cell outputs'

# -- OpenGraph options -------------------------------------------------------
#

ogp_site_url = "https://github.com/pyc-team/pytorch_concepts"
ogp_image = ogp_site_url + "_static/img/logos/pyc.png"

# -- Hoverxref options -------------------------------------------------------
#

hoverxref_auto_ref = True
hoverxref_roles = ['class', 'mod', 'doc', 'meth', 'func']
hoverxref_mathjax = True
hoverxref_intersphinx = ['PyG', 'numpy']

# -- Setup options -----------------------------------------------------------
#


def logo_role(name, rawtext, text, *args, **kwargs):
    if name == 'pyc':
        url = f'{html_baseurl}/_static/img/logos/pyc.svg'
    elif name == 'hydra':
        url = f'{html_baseurl}/_static/img/logos/hydra-head.svg'
    elif name in ['pyg', 'pytorch', 'lightning']:
        url = f'{html_baseurl}/_static/img/logos/{name}.svg'
    else:
        raise RuntimeError
    node = nodes.image(uri=url, alt=str(name).capitalize() + ' logo')
    node['classes'] += ['inline-logo', name]
    if text != 'null':
        node['classes'].append('with-text')
        span = nodes.inline(text=text)
        return [node, span], []
    return [node], []


def setup(app):

    def rst_jinja_render(app, docname, source):
        src = source[0]
        rendered = app.builder.templates.render_string(src, rst_context)
        source[0] = rendered

    app.connect("source-read", rst_jinja_render)

    app.add_role('pyc', logo_role)
    app.add_role('pyg', logo_role)
    app.add_role('pytorch', logo_role)
    app.add_role('hydra', logo_role)
    app.add_role('lightning', logo_role)
