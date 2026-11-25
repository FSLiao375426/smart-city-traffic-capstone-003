"""Local Flask mock serving the saved candidate; no fitting or outcome access."""
import logging
import time

from flask import Flask, jsonify, request
from werkzeug.exceptions import BadRequest, UnsupportedMediaType

from part3_machine_learning.recommend import BASE, load_bundle, recommend

logger = logging.getLogger(__name__)


def create_app(bundle=None):
    bundle = load_bundle() if bundle is None else bundle
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 16384

    @app.get('/health')
    def health():
        return jsonify(status='ready', model_run_id=bundle['model_run_id'],
                       mode='local_candidate_mock_not_final_test_approved')

    @app.post('/predict')
    def predict():
        started = time.perf_counter()
        try:
            result = recommend(request.get_json(), bundle)
        except (ValueError, TypeError, BadRequest, UnsupportedMediaType) as exc:
            logger.warning('Rejected prediction request: %s', exc)
            return jsonify(error='invalid_request', detail=str(exc)), 400
        logger.info('Prediction status=%s hours=%d elapsed_ms=%.3f model_run_id=%s',
                    result['status'], len(result['hourly_estimates']),
                    (time.perf_counter()-started)*1000, bundle['model_run_id'])
        return jsonify(result)

    return app


if __name__ == '__main__':
    from part2_python.pipeline import configure_logging
    configure_logging(BASE / 'serve.log')
    create_app().run(host='127.0.0.1', port=8000, debug=False)
