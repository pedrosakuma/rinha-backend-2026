// Wrapper around the upstream test.js that allows configurable duration / target rate
// via env vars (RUN_DURATION, RUN_TARGET, RESULT_FILE) so the same script powers
// both the short iteration loop and the full official-style run.
import http from 'k6/http';
import { SharedArray } from 'k6/data';
import { Counter } from 'k6/metrics';
import exec from 'k6/execution';

const RUN_DURATION = __ENV.RUN_DURATION || '30s';
const RUN_TARGET   = parseInt(__ENV.RUN_TARGET || '900', 10);
const RESULT_FILE  = __ENV.RESULT_FILE || 'bench/results/last.json';

const testData = new SharedArray('test-data', () => JSON.parse(open('./test-data.json')).entries);
const statsArr = new SharedArray('test-stats', () => [JSON.parse(open('./test-data.json')).stats]);
const expectedStats = statsArr[0];

const tpCount = new Counter('tp_count');
const tnCount = new Counter('tn_count');
const fpCount = new Counter('fp_count');
const fnCount = new Counter('fn_count');
const errorCount = new Counter('error_count');

export const options = {
    summaryTrendStats: ['p(50)', 'p(90)', 'p(99)', 'max'],
    systemTags: ['status', 'method'],
    scenarios: {
        default: {
            executor: 'ramping-arrival-rate',
            startRate: 1,
            timeUnit: '1s',
            preAllocatedVUs: 100,
            maxVUs: 250,
            gracefulStop: '5s',
            stages: [{ duration: RUN_DURATION, target: RUN_TARGET }],
        },
    },
};

export default function () {
    const idx = exec.scenario.iterationInTest % testData.length;
    const entry = testData[idx];
    const expectedApproved = entry.expected_approved;

    const res = http.post(
        'http://localhost:9999/fraud-score',
        JSON.stringify(entry.request),
        { headers: { 'Content-Type': 'application/json' }, timeout: '2001ms' }
    );

    if (res.status === 200) {
        const body = JSON.parse(res.body);
        if (expectedApproved === body.approved) {
            if (body.approved) tnCount.add(1);
            else tpCount.add(1);
        } else {
            if (body.approved) fnCount.add(1);
            else fpCount.add(1);
        }
    } else {
        errorCount.add(1);
    }
}

export function handleSummary(data) {
    const K = 1000;
    const T_MAX_MS = 1000;
    const P99_MIN_MS = 1;
    const P99_MAX_MS = 2000;
    const EPSILON_MIN = 0.001;
    const BETA = 300;
    const TX_CORTE = 0.15;

    const p99 = data.metrics.http_req_duration.values['p(99)'];
    const p50 = data.metrics.http_req_duration.values['p(50)'];
    const p90 = data.metrics.http_req_duration.values['p(90)'];
    const max = data.metrics.http_req_duration.values['max'];

    const tp = data.metrics.tp_count?.values.count || 0;
    const tn = data.metrics.tn_count?.values.count || 0;
    const fp = data.metrics.fp_count?.values.count || 0;
    const fn = data.metrics.fn_count?.values.count || 0;
    const errs = data.metrics.error_count?.values.count || 0;
    const N = tp + tn + fp + fn + errs;

    const E = (fp * 1) + (fn * 3) + (errs * 5);
    const failures = fp + fn + errs;
    const epsilon = N > 0 ? E / N : 0;
    const failureRate = N > 0 ? failures / N : 0;

    let p99Score, p99Cut = false;
    if (p99 <= 0) p99Score = 0;
    else if (p99 > P99_MAX_MS) { p99Score = -3000; p99Cut = true; }
    else p99Score = K * Math.log10(T_MAX_MS / Math.max(p99, P99_MIN_MS));

    let detScore, rateComp = 0, absPen = 0, detCut = false;
    if (failureRate > TX_CORTE) { detScore = -3000; detCut = true; }
    else {
        rateComp = K * Math.log10(1 / Math.max(epsilon, EPSILON_MIN));
        absPen = -BETA * Math.log10(1 + E);
        detScore = rateComp + absPen;
    }

    const finalScore = p99Score + detScore;

    const result = {
        run: {
            duration: RUN_DURATION,
            target_rps: RUN_TARGET,
            requests_total: N,
            iterations: data.metrics.iterations?.values.count || 0,
            actual_rps: data.metrics.iterations?.values.rate || 0,
        },
        latency: {
            p50: +p50.toFixed(2),
            p90: +p90.toFixed(2),
            p99: +p99.toFixed(2),
            max: +max.toFixed(2),
        },
        p99: p99.toFixed(2) + 'ms',
        scoring: {
            breakdown: { tp, tn, fp, fn, http_errors: errs },
            failure_rate: +(failureRate * 100).toFixed(2) + '%',
            weighted_errors_E: E,
            error_rate_epsilon: +epsilon.toFixed(6),
            p99_score: { value: +p99Score.toFixed(2), cut_triggered: p99Cut },
            detection_score: {
                value: +detScore.toFixed(2),
                rate_component: detCut ? null : +rateComp.toFixed(2),
                absolute_penalty: detCut ? null : +absPen.toFixed(2),
                cut_triggered: detCut,
            },
            final_score: +finalScore.toFixed(2),
        },
        expected: expectedStats,
    };

    const out = {};
    out[RESULT_FILE] = JSON.stringify(result, null, 2);
    return out;
}
