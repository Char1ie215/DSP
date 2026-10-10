from smoke_test import summarize


def test_missing_episode_is_not_a_pass():
    assert not summarize({}, False)['execution_passed']


def test_task_failure_is_distinct_from_execution_failure():
    result = summarize({'episode_outcomes': [{'success': False}], 'mean_success_rate': 0}, False)
    assert result['execution_passed']
    assert result['success_rate'] == 0


def test_perturbed_run_must_actually_apply_a_push():
    data = {'episode_outcomes': [{'success': False}]}
    assert not summarize(data, True)['execution_passed']
    data['concurrent_environment_perturbation'] = {'episode_records': [{'force_steps': 2}]}
    assert summarize(data, True)['execution_passed']


def test_recovery_coverage_is_explicit():
    data = {'episode_outcomes': [{}], 'executor': {'field_recovery_steps': 49,
            'recovery_reentry_replans': 1}}
    result = summarize(data, False)
    assert result['field_recovery_steps'] == 49
    assert result['recovery_reentry_replans'] == 1
