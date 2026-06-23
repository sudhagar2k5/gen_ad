#!/bin/bash
###############################################################################
# run_full_evaluation.sh
#
# Bench2Drive closed-loop evaluation, with a full preflight requirements check.
#
#   Phase 1  PREFLIGHT : conda env, python/torch/mmcv/carla, GPU, Vulkan,
#                        CARLA install, repos, checkpoints, routes, resources.
#   Phase 2  EVALUATE  : run the leaderboard closed-loop eval (CARLA is started
#                        by the evaluator itself).
#
# Usage:
#   ./run_full_evaluation.sh                 # check, then run (single route)
#   ./run_full_evaluation.sh --check-only    # only the preflight checks
#   ./run_full_evaluation.sh --full          # run all 220 routes
#   ./run_full_evaluation.sh --model genad   # use GenAD config+ckpt instead of VAD
#   ./run_full_evaluation.sh --force         # run even if checks FAIL
#
# Auto-fix (Phase 0, run before the preflight):
#   ./run_full_evaluation.sh --clone-repos   # git clone Bench2Drive(+Zoo) if missing
#   ./run_full_evaluation.sh --setup-env     # create the conda env from env/*.yml
#   ./run_full_evaluation.sh --deps-help     # print CARLA + checkpoint download steps
#   (combine, e.g.:  ./run_full_evaluation.sh --clone-repos --setup-env)
#
# Override paths via env vars (defaults shown):
#   CONDA_ROOT=/home/sudhagar/miniconda3
#   ENV_NAME=b2d_zoo
#   BENCH2DRIVE=/home/sudhagar/Bench2Drive
#   BENCH2DRIVEZOO=/home/sudhagar/Bench2DriveZoo
#   CARLA_ROOT=/home/sudhagar/carla
#   ENV_FILE=<path to b2d_zoo_environment.yml>   (auto-detected next to script)
#   B2D_REPO=https://github.com/Thinklab-SJTU/Bench2Drive.git
#   B2DZOO_REPO=https://github.com/Thinklab-SJTU/Bench2DriveZoo.git  (branch uniad/vad)
###############################################################################
set -uo pipefail

# ----------------------------- configuration ------------------------------- #
CONDA_ROOT="${CONDA_ROOT:-/home/sudhagar/miniconda3}"
ENV_NAME="${ENV_NAME:-b2d_zoo}"
BENCH2DRIVE="${BENCH2DRIVE:-/home/sudhagar/Bench2Drive}"
BENCH2DRIVEZOO="${BENCH2DRIVEZOO:-/home/sudhagar/Bench2DriveZoo}"
CARLA_ROOT="${CARLA_ROOT:-/home/sudhagar/carla}"

MODEL="vad"               # vad | genad
ROUTE_MODE="single"       # single | full
CHECK_ONLY=0
FORCE=0
SETUP_ENV=0
CLONE_REPOS=0
DEPS_HELP=0
PORT="${PORT:-30000}"
TM_PORT="${TM_PORT:-50000}"
GPU_RANK="${GPU_RANK:-0}"

# repos + env file for the auto-fix phase
B2D_REPO="${B2D_REPO:-https://github.com/Thinklab-SJTU/Bench2Drive.git}"
B2DZOO_REPO="${B2DZOO_REPO:-https://github.com/Thinklab-SJTU/Bench2DriveZoo.git}"
B2DZOO_BRANCH="${B2DZOO_BRANCH:-uniad/vad}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# auto-detect the exported env file shipped in the gen_ad repo
ENV_FILE="${ENV_FILE:-}"
if [ -z "$ENV_FILE" ]; then
  for cand in "$SCRIPT_DIR/env/b2d_zoo_environment.yml" \
              "$SCRIPT_DIR/env/b2d_zoo_environment_nobuilds.yml" \
              "./env/b2d_zoo_environment.yml"; do
    [ -f "$cand" ] && { ENV_FILE="$cand"; break; }
  done
fi

# ----------------------------- arg parsing --------------------------------- #
while [ $# -gt 0 ]; do
  case "$1" in
    --check-only) CHECK_ONLY=1 ;;
    --full)       ROUTE_MODE="full" ;;
    --single)     ROUTE_MODE="single" ;;
    --model)      MODEL="$2"; shift ;;
    --force)      FORCE=1 ;;
    --setup-env)  SETUP_ENV=1 ;;
    --clone-repos) CLONE_REPOS=1 ;;
    --deps-help)  DEPS_HELP=1 ;;
    -h|--help)    grep '^#' "$0" | sed 's/^#//'; exit 0 ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
  shift
done

# ----------------------------- pretty output ------------------------------- #
if [ -t 1 ]; then G="\033[32m"; R="\033[31m"; Y="\033[33m"; B="\033[1m"; N="\033[0m"; else G=""; R=""; Y=""; B=""; N=""; fi
PASS=0; WARN=0; FAILN=0
ok()   { echo -e "  ${G}[PASS]${N} $1"; PASS=$((PASS+1)); }
warn() { echo -e "  ${Y}[WARN]${N} $1"; WARN=$((WARN+1)); }
fail() { echo -e "  ${R}[FAIL]${N} $1"; FAILN=$((FAILN+1)); }
section() { echo -e "\n${B}== $1 ==${N}"; }

echo -e "${B}#################################################################${N}"
echo -e "${B}#  Bench2Drive closed-loop : preflight + evaluation             #${N}"
echo -e "${B}#################################################################${N}"
echo "  model=$MODEL  route_mode=$ROUTE_MODE  check_only=$CHECK_ONLY  force=$FORCE"

# ----------------------------- deps download help -------------------------- #
print_deps_help() {
  section "Manual downloads (CARLA + checkpoints) — cannot be auto-fetched safely"
  cat <<EOF
  CARLA 0.9.15  ->  $CARLA_ROOT
    wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz
    mkdir -p "$CARLA_ROOT" && tar -xzf CARLA_0.9.15.tar.gz -C "$CARLA_ROOT"
    # (also: AdditionalMaps_0.9.15.tar.gz for Town11/12/13/15)

  Checkpoints  ->  $BENCH2DRIVEZOO/ckpts/
    See the Bench2DriveZoo README "Pretrained Weights" section. Required:
      ckpts/vad_b2d_base.pth              (VAD closed-loop)
      ckpts/resnet50-19c8e357.pth         (backbone init)
      ckpts/GenAD/checkpoints.pth         (only for --model genad)
EOF
}
if [ "$DEPS_HELP" -eq 1 ]; then print_deps_help; exit 0; fi

# ----------------------------- Phase 0: auto-fix --------------------------- #
if [ "$CLONE_REPOS" -eq 1 ] || [ "$SETUP_ENV" -eq 1 ]; then
  section "Phase 0: auto-fix"

  if [ "$CLONE_REPOS" -eq 1 ]; then
    if [ -d "$BENCH2DRIVE/.git" ]; then
      echo -e "  ${G}[skip]${N} Bench2Drive already present at $BENCH2DRIVE"
    else
      echo "  cloning Bench2Drive -> $BENCH2DRIVE"
      git clone "$B2D_REPO" "$BENCH2DRIVE" && echo -e "  ${G}[ok]${N} Bench2Drive cloned" \
        || echo -e "  ${R}[err]${N} Bench2Drive clone failed"
    fi
    if [ -d "$BENCH2DRIVEZOO/.git" ]; then
      echo -e "  ${G}[skip]${N} Bench2DriveZoo already present at $BENCH2DRIVEZOO"
    else
      echo "  cloning Bench2DriveZoo ($B2DZOO_BRANCH) -> $BENCH2DRIVEZOO"
      git clone -b "$B2DZOO_BRANCH" "$B2DZOO_REPO" "$BENCH2DRIVEZOO" \
        && echo -e "  ${G}[ok]${N} Bench2DriveZoo cloned" \
        || echo -e "  ${R}[err]${N} Bench2DriveZoo clone failed"
    fi
    # convenience symlink expected by the launchers (Bench2Drive/Bench2DriveZoo)
    [ -e "$BENCH2DRIVE/Bench2DriveZoo" ] || ln -s "$BENCH2DRIVEZOO" "$BENCH2DRIVE/Bench2DriveZoo" 2>/dev/null
  fi

  if [ "$SETUP_ENV" -eq 1 ]; then
    if [ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
      echo -e "  ${R}[err]${N} conda not found at $CONDA_ROOT - install miniconda first"
    else
      # shellcheck disable=SC1091
      source "$CONDA_ROOT/etc/profile.d/conda.sh"
      if conda env list | grep -qE "/${ENV_NAME}\$|/${ENV_NAME}[[:space:]]|^${ENV_NAME}[[:space:]]"; then
        echo -e "  ${G}[skip]${N} conda env '$ENV_NAME' already exists"
      elif [ -z "$ENV_FILE" ] || [ ! -f "$ENV_FILE" ]; then
        echo -e "  ${R}[err]${N} env file not found (set ENV_FILE=path to b2d_zoo_environment.yml)"
      else
        echo "  creating conda env '$ENV_NAME' from $ENV_FILE (this can take several minutes)"
        conda env create -n "$ENV_NAME" -f "$ENV_FILE" \
          && echo -e "  ${G}[ok]${N} env '$ENV_NAME' created" \
          || echo -e "  ${R}[err]${N} env create failed - try the *_nobuilds.yml file"
      fi
    fi
  fi
  echo "  (note: CARLA + checkpoints are NOT auto-downloaded; run --deps-help for steps)"
fi

# ----------------------------- 1. conda env -------------------------------- #
section "1. Conda environment"
if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
  ok "conda found at $CONDA_ROOT"
  # shellcheck disable=SC1091
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
  if conda env list | grep -qE "^${ENV_NAME}\s|/${ENV_NAME}\$|/${ENV_NAME}\s"; then
    ok "conda env '$ENV_NAME' exists"
    conda activate "$ENV_NAME" && ok "activated '$ENV_NAME'" || fail "could not activate '$ENV_NAME'"
  else
    fail "conda env '$ENV_NAME' NOT found (recreate from env/b2d_zoo_environment.yml)"
  fi
else
  fail "conda not found at $CONDA_ROOT (set CONDA_ROOT=...)"
fi

# ----------------------------- 2. python packages -------------------------- #
section "2. Python packages"
PYV=$(python --version 2>&1); ok "python: $PYV"
pychk() {  # pychk <import> <label>
  if python -c "import $1" >/dev/null 2>&1; then
    v=$(python -c "import $1 as m; print(getattr(m,'__version__',''))" 2>/dev/null)
    ok "$2 importable ${v:+($v)}"
  else
    fail "$2 NOT importable (python -c 'import $1')"
  fi
}
python -c "import torch; print(torch.__version__, torch.cuda.is_available())" >/dev/null 2>&1 \
  && ok "torch: $(python -c 'import torch; print(torch.__version__, "cuda="+str(torch.cuda.is_available()))')" \
  || fail "torch not importable"
pychk numpy numpy
pychk cv2 opencv
pychk carla "carla python api"
# vendored mmcv needs Bench2DriveZoo on path
( cd "$BENCH2DRIVEZOO" 2>/dev/null && pychk mmcv "mmcv (vendored)" )

# ----------------------------- 3. GPU / CUDA ------------------------------- #
section "3. GPU / CUDA"
if command -v nvidia-smi >/dev/null 2>&1; then
  gpu=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | head -1)
  ok "nvidia-smi: $gpu"
else
  fail "nvidia-smi not found (no NVIDIA driver?)"
fi

# ----------------------------- 4. Vulkan (CRITICAL) ------------------------ #
section "4. Vulkan (CARLA rendering)"
if command -v vulkaninfo >/dev/null 2>&1; then
  vk=$(vulkaninfo --summary 2>/dev/null | grep -i deviceName | head -3)
  if echo "$vk" | grep -qi "llvmpipe"; then
    if echo "$vk" | grep -qivE "llvmpipe"; then
      ok "hardware Vulkan device present"
    else
      fail "Vulkan only has llvmpipe (software) -> CARLA RenderThread will crash. Need a hardware GPU Vulkan driver (native Linux NVIDIA, or dzn)."
    fi
  elif [ -n "$vk" ]; then
    ok "Vulkan device(s): $(echo "$vk" | tr '\n' ';')"
  else
    fail "vulkaninfo returned no device (Vulkan broken)"
  fi
else
  warn "vulkaninfo not installed (cannot verify GPU rendering; install vulkan-tools)"
fi

# ----------------------------- 5. CARLA ------------------------------------ #
section "5. CARLA simulator"
if [ -x "$CARLA_ROOT/CarlaUE4.sh" ] || [ -f "$CARLA_ROOT/CarlaUE4.sh" ]; then
  ok "CarlaUE4.sh present at $CARLA_ROOT"
else
  fail "CarlaUE4.sh missing at $CARLA_ROOT (set CARLA_ROOT=...)"
fi

# ----------------------------- 6. Repositories ----------------------------- #
section "6. Repositories"
for f in \
  "$BENCH2DRIVE/leaderboard/leaderboard/leaderboard_evaluator.py" \
  "$BENCH2DRIVE/leaderboard/scripts/run_evaluation.sh" \
  "$BENCH2DRIVE/leaderboard/team_code/vad_b2d_agent.py" \
  "$BENCH2DRIVE/scenario_runner" \
  "$BENCH2DRIVEZOO/mmcv" \
  "$BENCH2DRIVEZOO/adzoo/vad/configs/VAD/VAD_base_e2e_b2d.py" ; do
  [ -e "$f" ] && ok "found ${f#$HOME/}" || fail "missing $f"
done

# ----------------------------- 7. Checkpoints ------------------------------ #
section "7. Checkpoints"
need_ckpts=( "ckpts/vad_b2d_base.pth" "ckpts/resnet50-19c8e357.pth" )
[ "$MODEL" = "genad" ] && need_ckpts+=( "ckpts/GenAD/checkpoints.pth" )
for c in "${need_ckpts[@]}"; do
  p="$BENCH2DRIVEZOO/$c"
  if [ -f "$p" ]; then ok "$c ($(du -h "$p" | cut -f1))"; else fail "missing $c"; fi
done

# ----------------------------- 8. Routes ----------------------------------- #
section "8. Route files"
if [ "$ROUTE_MODE" = "full" ]; then ROUTE_FILE="leaderboard/data/bench2drive220.xml"; else ROUTE_FILE="leaderboard/data/route_single.xml"; fi
if [ -f "$BENCH2DRIVE/$ROUTE_FILE" ]; then
  ok "route file: $ROUTE_FILE"
else
  if [ "$ROUTE_MODE" = "single" ] && [ -f "$BENCH2DRIVE/leaderboard/data/bench2drive220.xml" ]; then
    warn "route_single.xml missing; will generate from bench2drive220.xml at run time"
  else
    fail "route file missing: $ROUTE_FILE"
  fi
fi

# ----------------------------- 9. Resources -------------------------------- #
section "9. System resources"
ram=$(free -g | awk '/Mem:/{print $2}'); avail=$(free -g | awk '/Mem:/{print $7}')
[ "${ram:-0}" -ge 12 ] && ok "RAM total ${ram}G (avail ${avail}G)" || warn "RAM total ${ram}G - large maps (Town12/13) may OOM"
disk=$(df -BG "$CARLA_ROOT" 2>/dev/null | awk 'NR==2{print $4}')
ok "disk free on CARLA volume: ${disk:-?}"

# ----------------------------- summary ------------------------------------- #
echo -e "\n${B}== Preflight summary ==${N}"
echo -e "  ${G}PASS=$PASS${N}  ${Y}WARN=$WARN${N}  ${R}FAIL=$FAILN${N}"

if [ "$FAILN" -gt 0 ] && [ "$FORCE" -ne 1 ]; then
  echo -e "${R}Preflight FAILED ($FAILN). Fix the above or re-run with --force.${N}"
  exit 1
fi
if [ "$CHECK_ONLY" -eq 1 ]; then
  echo -e "${G}Check-only mode: all required checks passed.${N}"
  exit 0
fi

# ============================ Phase 2: EVALUATE ============================= #
section "Phase 2: running closed-loop evaluation"

# regenerate single route if needed
if [ "$ROUTE_MODE" = "single" ] && [ ! -f "$BENCH2DRIVE/$ROUTE_FILE" ]; then
  python - "$BENCH2DRIVE" <<'PYEOF'
import sys, xml.etree.ElementTree as ET, os
b2d = sys.argv[1]
t = ET.parse(os.path.join(b2d,"leaderboard/data/bench2drive220.xml")); r = t.getroot()
routes = r.findall("route")
pref = ["Town01","Town02","Town03","Town04","Town05","Town06","Town07","Town10HD"]
chosen = next((x for town in pref for x in routes if x.get("town")==town), routes[0])
for x in list(routes):
    if x is not chosen: r.remove(x)
ET.ElementTree(r).write(os.path.join(b2d,"leaderboard/data/route_single.xml"), encoding="utf-8", xml_declaration=True)
print("generated route_single.xml ->", chosen.get("id"), chosen.get("town"))
PYEOF
fi

export CARLA_ROOT
export CARLA_SERVER="${CARLA_ROOT}/CarlaUE4.sh"
export PYTHONPATH="$PYTHONPATH:${CARLA_ROOT}/PythonAPI:${CARLA_ROOT}/PythonAPI/carla:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg"

cd "$BENCH2DRIVE" || exit 1

if [ "$MODEL" = "genad" ]; then
  TEAM_CONFIG="Bench2DriveZoo/adzoo/genad/configs/VAD/GenAD_config_b2d.py+Bench2DriveZoo/ckpts/GenAD/checkpoints.pth"
  ENDPOINT="eval_genad.json"; SAVE="./eval_genad_closedloop/"
else
  TEAM_CONFIG="Bench2DriveZoo/adzoo/vad/configs/VAD/VAD_base_e2e_b2d.py+Bench2DriveZoo/ckpts/vad_b2d_base.pth"
  ENDPOINT="eval_vad.json"; SAVE="./eval_vad_closedloop/"
fi
TEAM_AGENT="leaderboard/team_code/vad_b2d_agent.py"

echo "  ROUTES=$ROUTE_FILE"
echo "  AGENT=$TEAM_AGENT"
echo "  CONFIG=$TEAM_CONFIG"
echo "  ENDPOINT=$ENDPOINT  SAVE=$SAVE"

bash leaderboard/scripts/run_evaluation.sh \
  "$PORT" "$TM_PORT" "True" "$ROUTE_FILE" \
  "$TEAM_AGENT" "$TEAM_CONFIG" "$ENDPOINT" \
  "$SAVE" "traj" "$GPU_RANK"
rc=$?

echo "  cleaning up stray CARLA..."
pkill -9 -f CarlaUE4 >/dev/null 2>&1

section "Results"
if [ -f "$BENCH2DRIVE/$ENDPOINT" ]; then
  ok "results written: $BENCH2DRIVE/$ENDPOINT"
  python - "$BENCH2DRIVE/$ENDPOINT" <<'PYEOF' 2>/dev/null || true
import sys, json
d = json.load(open(sys.argv[1]))
rec = d.get("_checkpoint", {}).get("records", [])
prog = d.get("_checkpoint", {}).get("progress", [])
print("  routes evaluated:", prog)
for r in rec[:10]:
    s = r.get("scores", {})
    print(f"   route {r.get('route_id')}: DS={s.get('score_composite')} "
          f"RC={s.get('score_route')} IS={s.get('score_penalty')} status={r.get('status')}")
PYEOF
else
  fail "no results file produced (exit rc=$rc) - check log above"
fi
echo -e "\n${B}Done (rc=$rc).${N}"
exit $rc
