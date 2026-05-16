import http from 'k6/http';
import { SharedArray } from 'k6/data';
import exec from 'k6/execution';

const RUN_DURATION = __ENV.RUN_DURATION || '45s';
const RUN_TARGET   = parseInt(__ENV.RUN_TARGET || '3000', 10);

const testData = new SharedArray('test-data', () =>
  JSON.parse(open('./test-data.json')).entries);

export const options = {
  summaryTrendStats: ['p(50)', 'p(90)', 'p(99)', 'p(99.9)', 'max'],
  scenarios: {
    constant: {
      executor: 'constant-arrival-rate',
      rate: RUN_TARGET,
      timeUnit: '1s',
      duration: RUN_DURATION,
      preAllocatedVUs: 300,
      maxVUs: 1000,
    },
  },
};

export default function () {
  const idx = exec.scenario.iterationInTest % testData.length;
  const entry = testData[idx];
  http.post('http://localhost:9999/fraud-score',
    JSON.stringify(entry.request),
    { headers: { 'Content-Type': 'application/json' }, timeout: '2001ms' });
}
