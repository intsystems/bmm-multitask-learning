import shutil


def add_notebook_example(*args, **kwargs):
    shutil.copy2(
        "examples/task_clustering/multi-task-learning.ipynb", 
        "docs/task_clustering/multi-task-learning.ipynb"
    )
