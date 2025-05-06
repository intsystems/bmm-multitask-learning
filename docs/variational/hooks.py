import shutil


def on_pre_build(*args, **kwargs):
    shutil.copy2(
        "examples/variational/elementary/elementary.ipynb", 
        "docs/variational/elementary.ipynb"
    )
