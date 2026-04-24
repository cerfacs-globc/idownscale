from pathlib import Path
import sys
sys.path.insert(0, str(Path('..').resolve()))

project = 'idownscale'
copyright = '2026, Zoé GARCIA'  # noqa: A001
author = 'Zoé GARCIA'
release = '0.1.1'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
