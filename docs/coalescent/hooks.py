import shutil


def on_pre_build(*args, **kwargs):
    shutil.copy2(
        "examples/coalescent/coalescent_example.ipynb", 
        "docs/coalescent/coalescent_example.ipynb"
    )
