"""Hour-distribution drift simulation; PASS does not establish model accuracy."""
import json
import logging
from numbers import Integral

import numpy as np

from part3_machine_learning.recommend import BASE, load_bundle

logger = logging.getLogger(__name__)
MIN_BATCH = 100
TV_THRESHOLD = 0.20


def hour_distribution(hours):
    values = list(hours)
    if not values or any(isinstance(v, bool) or not isinstance(v, Integral) or not 0 <= v < 24 for v in values):
        raise ValueError('Hours must be a nonempty sequence of integers in 0..23')
    return np.bincount(values, minlength=24) / len(values)


def check_hours(reference, current):
    reference, current = list(reference), list(current)
    p, q = hour_distribution(reference), hour_distribution(current)
    distance = float(np.abs(p-q).sum()/2)
    reasons = []
    if len(current) < MIN_BATCH:
        reasons.append('insufficient_batch')
    if distance > TV_THRESHOLD:
        reasons.append('hour_distribution_drift')
    result = dict(status='ALERT' if reasons else 'PASS', reasons=reasons,
                  feature='source_calendar_hour', reference_count=len(reference), current_count=len(current),
                  total_variation_distance=distance, threshold=TV_THRESHOLD, minimum_batch=MIN_BATCH,
                  reference_probabilities=p.tolist(), current_probabilities=q.tolist(),
                  threshold_basis='Analyst prototype setting; not statistically calibrated',
                  scope='Feature drift only; no observed traffic labels or accuracy conclusion')
    if reasons:
        logger.warning('Monitoring ALERT: %s TV=%.6f', reasons, distance)
    else:
        logger.info('Monitoring PASS TV=%.6f', distance)
    return result


def main():
    from part2_python.pipeline import configure_logging
    configure_logging(BASE / 'monitoring.log')
    bundle = load_bundle()
    reference = bundle['support_history'].hour.tolist()
    batches = {'reference_replay': reference, 'synthetic_shift': [8]*1000,
               'too_small': [8]*5}
    output = BASE / 'results/monitoring'
    output.mkdir(parents=True, exist_ok=True)
    evidence = {'model_run_id': bundle['model_run_id'], 'fit_end': bundle['fit_end'],
                'simulation_only': True, 'cases': {}}
    for name, hours in batches.items():
        evidence['cases'][name] = {'current_hours': hours, 'result': check_hours(reference, hours)}
    (output / 'simulation.json').write_text(json.dumps(evidence, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
