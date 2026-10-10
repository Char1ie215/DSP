from typing import List
from libero.libero.benchmark.libero_suite_task_map import libero_task_map

MT_TASKS = {
    'libero10': libero_task_map['libero_10'],
    'libero_10': libero_task_map['libero_10'],
    'libero90': libero_task_map['libero_90'],
    'libero_90': libero_task_map['libero_90'],
    'libero_spatial': libero_task_map['libero_spatial'],
    'libero_object': libero_task_map['libero_object'],
    'libero_goal': libero_task_map['libero_goal'],
}

def is_multitask(task_name: str) -> bool:
    return task_name in MT_TASKS


def get_subtasks(task_name: str) -> List[str]:
    if is_multitask(task_name):
        return MT_TASKS[task_name]
    else:
        return [task_name]
