import shutil


def on_pre_build(*args, **kwargs):
    shutil.copy2(
        "examples/task_clustering/multi-task-learning.ipynb", 
        "docs/task_clustering/multi-task-learning.ipynb"
    )
