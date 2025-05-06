import shutil


def add_notebook_example(*args, **kwargs):
    shutil.copy2(
        "examples/sbmtl/experiment_testing_method.ipynb", 
        "docs/sbmtl/experiment_testing_method.ipynb"
    )
