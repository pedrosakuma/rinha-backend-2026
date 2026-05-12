#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${OUT_DIR:-/tmp/rinha-prof/${TS}}"
PROJECT="${PROJECT:-rinha-prof-${TS,,}}"

PERF_IMAGE="${PERF_IMAGE:-rinha-perf-sidecar:local}"
PERF_TOOLS_VERSION="${PERF_TOOLS_VERSION:-6.8.0-111}"
PERF_EVENT="${PERF_EVENT:-task-clock}"
PERF_FREQ="${PERF_FREQ:-997}"
PERF_CALLGRAPH="${PERF_CALLGRAPH:-fp}"
PERF_SCOPE="${PERF_SCOPE:-system}"
PERF_CGROUP="${PERF_CGROUP:-}"
PROFILE_SECONDS="${PROFILE_SECONDS:-45}"
PROFILE_TARGETS="${PROFILE_TARGETS:-api,lb}"
PERF_ARM_DELAY_SECONDS="${PERF_ARM_DELAY_SECONDS:-2}"
READY_URL="${READY_URL:-http://localhost:9999/ready}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-90}"
POST_READY_SLEEP_SECONDS="${POST_READY_SLEEP_SECONDS:-2}"

K6_DURATION="${K6_DURATION:-${PROFILE_SECONDS}s}"
K6_TARGET="${K6_TARGET:-900}"
K6_SCRIPT="${K6_SCRIPT:-bench/k6/test.runner.js}"
WARMUP_K6_DURATION="${WARMUP_K6_DURATION:-3s}"
WARMUP_K6_TARGET="${WARMUP_K6_TARGET:-${K6_TARGET}}"

BUILD_PERF_IMAGE="${BUILD_PERF_IMAGE:-1}"
BUILD_API_IMAGE="${BUILD_API_IMAGE:-0}"
PROFILE_API_IMAGE="${PROFILE_API_IMAGE:-rinha-api-profile:local}"
API_IMAGE="${API_IMAGE:-}"
API_PULL_POLICY="${API_PULL_POLICY:-never}"
API_STRIP_SYMBOLS="${API_STRIP_SYMBOLS:-false}"
IVF_NLIST_BUILD="${IVF_NLIST_BUILD:-1024}"
IVF_BAL_SLACK_BUILD="${IVF_BAL_SLACK_BUILD:-0}"
IVF_HEAVY_SPLIT_MAX="${IVF_HEAVY_SPLIT_MAX:-0}"
EXTRA_COMPOSE_FILE="${EXTRA_COMPOSE_FILE:-}"

mkdir -p "$OUT_DIR"

run_k6() {
  local duration="$1"
  local target="$2"
  local result_file="$3"
  local log_file="$4"

  (
    cd "$(dirname "$K6_SCRIPT")"
    RUN_DURATION="$duration" \
    RUN_TARGET="$target" \
    RESULT_FILE="$result_file" \
    k6 run --quiet "$(basename "$K6_SCRIPT")"
  ) >"$log_file" 2>&1
}

if [[ "$BUILD_PERF_IMAGE" == "1" ]]; then
  docker build -q \
    -f docker/Dockerfile.perf-sidecar \
    --build-arg "PERF_TOOLS_VERSION=${PERF_TOOLS_VERSION}" \
    -t "$PERF_IMAGE" . >/dev/null
fi

if [[ "$BUILD_API_IMAGE" == "1" ]]; then
  docker build -q \
    -f docker/Dockerfile.api \
    --build-arg "API_STRIP_SYMBOLS=${API_STRIP_SYMBOLS}" \
    --build-arg "IVF_NLIST_BUILD=${IVF_NLIST_BUILD}" \
    --build-arg "IVF_BAL_SLACK_BUILD=${IVF_BAL_SLACK_BUILD}" \
    --build-arg "IVF_HEAVY_SPLIT_MAX=${IVF_HEAVY_SPLIT_MAX}" \
    -t "$PROFILE_API_IMAGE" . >/dev/null
  API_IMAGE="$PROFILE_API_IMAGE"
fi

compose_files=(-f docker-compose.yml)
override_file=""
if [[ -n "$API_IMAGE" ]]; then
  override_file="${OUT_DIR}/compose.override.yml"
  cat > "$override_file" <<YAML
services:
  api1:
    image: ${API_IMAGE}
    pull_policy: ${API_PULL_POLICY}
  api2:
    image: ${API_IMAGE}
    pull_policy: ${API_PULL_POLICY}
YAML
  compose_files+=(-f "$override_file")
fi
if [[ -n "$EXTRA_COMPOSE_FILE" ]]; then
  compose_files+=(-f "$EXTRA_COMPOSE_FILE")
fi

cleanup() {
  docker compose -p "$PROJECT" "${compose_files[@]}" down -v >"${OUT_DIR}/compose-down.log" 2>&1 || true
}
trap cleanup EXIT

echo ">> out: ${OUT_DIR}"
echo ">> project: ${PROJECT}"
docker compose -p "$PROJECT" "${compose_files[@]}" up -d --wait >"${OUT_DIR}/compose-up.log"

echo ">> waiting for ${READY_URL}"
ready_deadline=$((SECONDS + READY_TIMEOUT_SECONDS))
until curl --fail --silent --max-time 1 "$READY_URL" >/dev/null; do
  if (( SECONDS >= ready_deadline )); then
    echo "timed out waiting for ${READY_URL}" >&2
    docker compose -p "$PROJECT" "${compose_files[@]}" logs --no-color >"${OUT_DIR}/compose.log" || true
    exit 1
  fi
  sleep 1
done
sleep "$POST_READY_SLEEP_SECONDS"

if [[ "$WARMUP_K6_DURATION" != "0" && "$WARMUP_K6_DURATION" != "0s" ]]; then
  echo ">> warmup k6: duration=${WARMUP_K6_DURATION} target=${WARMUP_K6_TARGET}"
  run_k6 "$WARMUP_K6_DURATION" "$WARMUP_K6_TARGET" "${OUT_DIR}/k6-warmup-result.json" "${OUT_DIR}/k6-warmup.log"
fi

declare -a services=()
if [[ ",${PROFILE_TARGETS}," == *",api,"* ]]; then
  services+=(api1 api2)
fi
if [[ ",${PROFILE_TARGETS}," == *",lb,"* ]]; then
  services+=(lb)
fi

declare -a pids=()
declare -a cids=()
: >"${OUT_DIR}/pids.txt"
for service in "${services[@]}"; do
  cid="$(docker compose -p "$PROJECT" "${compose_files[@]}" ps -q "$service")"
  pid="$(docker inspect -f '{{.State.Pid}}' "$cid")"
  cids+=("$cid")
  pids+=("$pid")
  printf '%s %s %s\n' "$service" "$cid" "$pid" | tee -a "${OUT_DIR}/pids.txt"
done

pid_csv="$(IFS=,; echo "${pids[*]}")"
if [[ -z "$pid_csv" ]]; then
  echo "no PIDs selected; PROFILE_TARGETS=${PROFILE_TARGETS}" >&2
  exit 1
fi

sidecar_base=(docker run --rm --privileged --pid=host --cgroupns=host)

mkdir -p "${OUT_DIR}/symfs/app"
for i in "${!services[@]}"; do
  case "${services[$i]}" in
    api1)
      docker cp "${cids[$i]}:/app/Rinha.Api" "${OUT_DIR}/symfs/app/Rinha.Api" >/dev/null 2>&1 || true
      ;;
    lb)
      docker cp "${cids[$i]}:/proxy" "${OUT_DIR}/symfs/proxy" >/dev/null 2>&1 || true
      ;;
  esac
done

tid_csv="$("${sidecar_base[@]}" "$PERF_IMAGE" \
  'set -euo pipefail
   IFS=, read -ra pids <<< "'"$pid_csv"'"
   for pid in "${pids[@]}"; do
     for task in /proc/"$pid"/task/*; do
       basename "$task"
     done
   done' | paste -sd, -)"
printf '%s\n' "${tid_csv//,/$'\n'}" >"${OUT_DIR}/tids.txt"

cgroup_csv="$("${sidecar_base[@]}" "$PERF_IMAGE" \
  'set -euo pipefail
   IFS=, read -ra pids <<< "'"$pid_csv"'"
   for pid in "${pids[@]}"; do
     while IFS=: read -r hierarchy _ path; do
       if [[ "$hierarchy" == "0" ]]; then
         path="${path#/}"
         printf "%s\n" "$path"
       fi
     done < /proc/"$pid"/cgroup
   done' | paste -sd, -)"
printf '%s\n' "${cgroup_csv//,/$'\n'}" >"${OUT_DIR}/cgroups.txt"
IFS=, read -ra cgroup_items <<< "$cgroup_csv"
if [[ -n "$PERF_CGROUP" ]]; then
  printf '%s\n' "$PERF_CGROUP" >"${OUT_DIR}/perf-cgroup.txt"
fi

case "$PERF_SCOPE" in
  cgroup) perf_record_args="" ;;
  tid) perf_record_args="-e ${PERF_EVENT} -F ${PERF_FREQ} -g --call-graph ${PERF_CALLGRAPH} -t ${tid_csv}" ;;
  system) perf_record_args="-e ${PERF_EVENT} -F ${PERF_FREQ} -g --call-graph ${PERF_CALLGRAPH} -a" ;;
  pid) perf_record_args="-e ${PERF_EVENT} -F ${PERF_FREQ} -g --call-graph ${PERF_CALLGRAPH} -p ${pid_csv}" ;;
  pid-system) perf_record_args="-a -e ${PERF_EVENT} -F ${PERF_FREQ} -g --call-graph ${PERF_CALLGRAPH} -p ${pid_csv}" ;;
  *)
    echo "invalid PERF_SCOPE=${PERF_SCOPE}; expected cgroup, tid, system, pid, or pid-system" >&2
    exit 1
    ;;
esac

"${sidecar_base[@]}" -v "${OUT_DIR}:/out" "$PERF_IMAGE" \
  '"$PERF_BIN" --version; uname -a; ps -L -p '"$pid_csv"' -o pid,tid,comm,args' \
  >"${OUT_DIR}/sidecar-env.txt" 2>&1 || true

echo ">> perf record: scope=${PERF_SCOPE} event=${PERF_EVENT} freq=${PERF_FREQ} callgraph=${PERF_CALLGRAPH} pids=${pid_csv}"
declare -a perf_pids=()
if [[ "$PERF_SCOPE" == "cgroup" ]]; then
  if [[ -n "$PERF_CGROUP" ]]; then
    cgroup_names=(custom)
    cgroup_values=("$PERF_CGROUP")
  else
    cgroup_names=("${services[@]}")
    cgroup_values=("${cgroup_items[@]}")
  fi

  for i in "${!cgroup_values[@]}"; do
    name="${cgroup_names[$i]}"
    cg="${cgroup_values[$i]}"
    echo ">> perf cgroup ${name}: ${cg}"
    "${sidecar_base[@]}" -v "${OUT_DIR}:/out" "$PERF_IMAGE" \
      'set -euo pipefail; "$PERF_BIN" record -a -e '"$PERF_EVENT"' -G '"$cg"' -F '"$PERF_FREQ"' -g --call-graph '"$PERF_CALLGRAPH"' -o /out/perf-'"$name"'.data -- sleep '"$PROFILE_SECONDS" \
      >"${OUT_DIR}/perf-record-${name}.log" 2>&1 &
    perf_pids+=("$!")
  done
else
  "${sidecar_base[@]}" -v "${OUT_DIR}:/out" "$PERF_IMAGE" \
    'set -euo pipefail; "$PERF_BIN" record '"$perf_record_args"' -o /out/perf.data -- sleep '"$PROFILE_SECONDS" \
    >"${OUT_DIR}/perf-record.log" 2>&1 &
  perf_pids+=("$!")
fi

sleep "$PERF_ARM_DELAY_SECONDS"

echo ">> k6: duration=${K6_DURATION} target=${K6_TARGET}"
set +e
run_k6 "$K6_DURATION" "$K6_TARGET" "${OUT_DIR}/k6-result.json" "${OUT_DIR}/k6.log"
k6_status=$?
perf_status=0
for pid in "${perf_pids[@]}"; do
  wait "$pid"
  status=$?
  if [[ "$status" != "0" ]]; then
    perf_status="$status"
  fi
done
set -e

if [[ "$PERF_SCOPE" == "cgroup" ]]; then
  : >"${OUT_DIR}/perf-report.txt"
  : >"${OUT_DIR}/perf-report-children.txt"
  for name in "${cgroup_names[@]}"; do
    "${sidecar_base[@]}" -v "${OUT_DIR}:/out" "$PERF_IMAGE" \
      'set -euo pipefail
       "$PERF_BIN" report --symfs /out/symfs --stdio --no-children --sort comm,dso,symbol -i /out/perf-'"$name"'.data > /out/perf-report-'"$name"'.txt || true
       "$PERF_BIN" report --symfs /out/symfs --stdio --children --sort comm,dso,symbol -i /out/perf-'"$name"'.data > /out/perf-report-children-'"$name"'.txt || true
       "$PERF_BIN" buildid-list -i /out/perf-'"$name"'.data > /out/buildids-'"$name"'.txt || true
       chown -R '"$(id -u):$(id -g)"' /out' \
      >"${OUT_DIR}/perf-report-${name}.log" 2>&1 || true
    {
      printf '===== %s =====\n' "$name"
      cat "${OUT_DIR}/perf-report-${name}.txt"
      printf '\n'
    } >>"${OUT_DIR}/perf-report.txt"
    {
      printf '===== %s =====\n' "$name"
      cat "${OUT_DIR}/perf-report-children-${name}.txt"
      printf '\n'
    } >>"${OUT_DIR}/perf-report-children.txt"
  done
else
  "${sidecar_base[@]}" -v "${OUT_DIR}:/out" "$PERF_IMAGE" \
    'set -euo pipefail
     "$PERF_BIN" report --symfs /out/symfs --stdio --no-children --sort comm,dso,symbol -i /out/perf.data > /out/perf-report.txt || true
     "$PERF_BIN" report --symfs /out/symfs --stdio --children --sort comm,dso,symbol -i /out/perf.data > /out/perf-report-children.txt || true
     "$PERF_BIN" report --symfs /out/symfs --stdio --no-children --comms ".NET TP Worker,.NET Sockets" --sort comm,dso,symbol -i /out/perf.data > /out/perf-report-dotnet.txt || true
     "$PERF_BIN" report --symfs /out/symfs --stdio --no-children --dsos Rinha.Api --sort comm,dso,symbol -i /out/perf.data > /out/perf-report-rinha-dso.txt || true
     "$PERF_BIN" report --symfs /out/symfs --stdio --no-children --comms proxy --sort comm,dso,symbol -i /out/perf.data > /out/perf-report-lb.txt || true
     "$PERF_BIN" buildid-list -i /out/perf.data > /out/buildids.txt || true
     chown -R '"$(id -u):$(id -g)"' /out' \
    >"${OUT_DIR}/perf-report.log" 2>&1 || true
fi

echo ">> perf status=${perf_status} k6 status=${k6_status}"
echo ">> artifacts:"
find "$OUT_DIR" -maxdepth 1 -type f -printf '   %f\n' | sort

if [[ "$perf_status" != "0" || "$k6_status" != "0" ]]; then
  exit 1
fi
