# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from datetime import date

import WFacer

project = "WFacer"
copyright = f"2022-{date.today().year}, Ceder Group"
author = "Fengyu Xie"
release = WFacer.__version__
version = WFacer.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    # "sphinx.ext.coverage",
    # "sphinx.ext.doctest",
    # "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx_mdinclude",
]

# Generate the API documentation when building
autosummary_generate = True
add_module_names = False
autoclass_content = "both"

napoleon_google_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_custom_sections = None

# Add any paths that contain templates here, relative to this directory.
source_suffix = [".rst"]

# The encoding of src files.
source_encoding = "utf-8"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to src directory, that match files and
# directories to ignore when looking for src files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = "pydata_sphinx_theme"

# TODO: update this when fixed version of pydata-sphinx-theme is released
# html_logo = "_static/logo.png"  # banner.svg needs text as paths to avoid font missing

html_theme_options = {
    "logo": {
        "text": "WFacer",
    },
    "github_url": "https://github.com/CederGroupHub/WFacer",
    "use_edit_page_button": True,
    "show_toc_level": 2,
    # "navbar_align": "left",  # [left, content, right] For testing that the navbar
    # items align properly
    "navbar_start": ["navbar-logo", "navbar-version"],
    # "navbar_center": ["navbar-nav", "navbar-version"],  # Just for testing
    "navigation_depth": 2,
    "show_nav_level": 2,
    "navbar_end": ["theme-switcher", "navbar-icon-links"],  #
    # "left_sidebar_end": ["custom-template.html", "sidebar-ethical-ads.html"],
    # "footer_items": ["copyright", "sphinx-version", ""]
    "external_links": [
        {
            "name": "Changes",
            "url": "https://github.com/CederGroupHub/WFacer/blob/master/CHANGES.md",
        },
        {"name": "Issues", "url": "https://github.com/CederGroupHub/WFacer/issues"},
    ],
}

html_context = {
    "github_url": "https://github.com",  # or your GitHub Enterprise interprise
    "github_user": "CederGroupHub",
    "github_repo": "WFacer",
    "github_version": "main",
    "doc_path": "docs/src",
    "source_suffix": source_suffix,
    "default_mode": "auto",
}

# Custom sidebar templates, maps page names to templates.
html_sidebars = {
    "contribute/index": [
        "search-field",
        "sidebar-nav-bs",
        "custom-template",
    ],  # This ensures we test for custom sidebars
    # "demo/no-sidebar": [],  # Test what page looks like with no sidebar items
}

# The style sheet to use for HTML and HTML Help pages. A file of that name
# must exist either in Sphinx' static/ path, or in one of the custom paths
# given in html_static_path.
# html_style = ''

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

html_static_path = ["_static"]

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = "%b %d, %Y"

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
# html_use_smartypants = True

# Content template for the index page.
html_index = "index.html"

# If false, no module index is generated.
html_use_modindex = True

html_file_suffix = ".html"

# If true, the reST sources are included in the HTML build as _sources/<name>.
html_copy_source = False

# Output file base name for HTML help builder.
htmlhelp_basename = "WFacer"
