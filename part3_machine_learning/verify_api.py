"""Exercise a real loopback HTTP server with a disk-loaded bundle, without fitting."""
import hashlib
import json
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
from werkzeug.serving import make_server

from part3_machine_learning.recommend import BASE, BUNDLE_PATH
from part3_machine_learning.serve import create_app


def main():
    from part2_python.pipeline import configure_logging
    configure_logging(BASE / 'verify_api.log')
    before = hashlib.sha256(BUNDLE_PATH.read_bytes()).hexdigest()
    server = make_server('127.0.0.1', 0, create_app())
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    cases = []
    try:
        def call(path, payload=None):
            body = None if payload is None else json.dumps(payload).encode('utf-8')
            req = Request(f'http://127.0.0.1:{server.server_port}{path}', data=body,
                          headers={'Content-Type': 'application/json'})
            try:
                response = urlopen(req, timeout=30)
            except HTTPError as exc:
                response = exc
            with response:
                return response.status, json.loads(response.read())
        code, result = call('/health')
        assert code == 200 and result['status'] == 'ready'
        cases.append(dict(name='health', http_status=code, response=result))
        for name in ('weekday_clear', 'weekend_clear', 'weekday_rain', 'rare_squall'):
            request = json.loads((BASE / f'results/recommendation/requests/{name}.json').read_text(encoding='utf-8'))
            expected = json.loads((BASE / f'results/recommendation/{name}.json').read_text(encoding='utf-8'))
            code, result = call('/predict', request)
            assert code == 200 and result['status'] == expected['status']
            np.testing.assert_allclose([x['estimated_traffic'] for x in result['hourly_estimates']],
                                       [x['estimated_traffic'] for x in expected['hourly_estimates']], atol=1e-8, rtol=0)
            assert result['best_window'] == expected['best_window']
            cases.append(dict(name=name, http_status=code, response=result))
        for payload in ({}, {'traffic_volume': 1}, []):
            code, result = call('/predict', payload)
            assert code == 400 and result['error'] == 'invalid_request'
            cases.append(dict(name='invalid_input', input=payload, http_status=code, response=result))
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
    assert hashlib.sha256(BUNDLE_PATH.read_bytes()).hexdigest() == before
    output = BASE / 'results/serving'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'http_acceptance.json').write_text(json.dumps(dict(
        transport='real loopback HTTP; ephemeral local server stopped after verification',
        bundle_sha256=before, training_performed=False, cases=cases), ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
