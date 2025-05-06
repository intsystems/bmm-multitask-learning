import shutil


def add_notebook_example(*args, **kwargs):
    shutil.copy2(
        "examples/variational/elementary/elementary.ipynb", 
        "docs/variational/elementary.ipynb"
    )
