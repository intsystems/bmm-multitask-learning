import shutil


def on_pre_build(*args, **kwargs):
    shutil.copy2(
        "examples/sbmtl/experiment_testing_method.ipynb", 
        "docs/sbmtl/experiment_testing_method.ipynb"
    )
