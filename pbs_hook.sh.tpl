#!/bin/bash
#PBS -N {{ job_name }}
#PBS -l nodes={{ nodes }}:ppn={{ ppn }}:c{{ ppn }}
#PBS -l walltime={{ walltime }}
#PBS -j oe
#PBS -q {{ queue }}
ulimit -s unlimited

#set -euo pipefail

# parameters
VER={{ VER }}          # version option   : 5.4.4 6.2.1
TYPE1={{ TYPE1 }}      # type1 option     : org beef vtst beefvtst
TYPE2={{ TYPE2 }}      # type2 option     : std gam ncl gpu (for gpu, use 5.4.4)
OPT={{ OPT }}          # optimization option: 2 3 (2 stable; 3 slightly faster than 2)
COMPILER={{ COMPILER }}  # select intel compiler: 2018u3 2020u2

if [ "${COMPILER}" = "2020" ] ; then
    IMPIVER=2019.7.217
elif [ "${COMPILER}" = "2018u3" ] ; then
    IMPIVER=2018.3.222
elif [ "${COMPILER}" = "2020u2" ] ; then
    IMPIVER=2019.8.254
else
    echo "[FATAL] Unknown COMPILER=$COMPILER"
    exit 2
fi

VASPHOME=/data/software/vasp/compile/
LOG_FILE=run.log

WORK_ROOT="{{ project_root }}"
PARAMS_FILE="{{ params_file }}"
STAGE="{{ stage }}"
WORKDIR="{{ workdir }}"

LOBSTER_BIN="{{ params.get('lobster', {}).get('bin', '') }}"
# load enviromental parameters
source /data/opt/intel${COMPILER}/compilers_and_libraries/linux/bin/compilervars.sh intel64
export LD_LIBRARY_PATH=/data/opt/intel${COMPILER}/mkl/lib/intel64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/data/opt/intel${COMPILER}/lib/intel64:$LD_LIBRARY_PATH
source /data/opt/intel${COMPILER}/impi/${IMPIVER}/intel64/bin/mpivars.sh

cd "${WORKDIR}"

is_lobster_stage() {
  case "${STAGE}" in
    *_lobster) return 0 ;;
    *) return 1 ;;
  esac
}
# BEEF family needs vdw_kernel.bindat
if [[ "${TYPE1}" == beef* ]]; then
  VDW_SRC="/data2/home/luodh/script/bulk/BF/vdw_kernel.bindat"
  VDW_DST="${WORKDIR}/vdw_kernel.bindat"

  if [ ! -f "${VDW_DST}" ]; then
    if [ ! -f "${VDW_SRC}" ]; then
      echo "[FATAL] TYPE1=${TYPE1} requires vdw_kernel.bindat, but missing source: ${VDW_SRC}" >&2
      exit 30
    fi
    ln -f "${VDW_SRC}" "${VDW_DST}" >/dev/null 2>&1 || cp -f "${VDW_SRC}" "${VDW_DST}" >/dev/null 2>&1
  fi
fi

# If already converged, skip VASP
SKIP_VASP=0
if [ -f OUTCAR ]; then
  if tail -n 80 OUTCAR | grep -qiE "total cpu time used|voluntary context switches"; then
    SKIP_VASP=1
  fi
fi

if [ "${SKIP_VASP}" -eq 1 ]; then
  echo "[INFO] OUTCAR indicates normal termination; skip VASP run." >&2
else
  mpirun $VASPHOME$VER\_$COMPILER/vasp.$VER\_$TYPE1\_O$OPT/bin/vasp_$TYPE2 > $LOG_FILE 2>&1
fi

# Hard fail if OUTCAR missing or not terminated normally
if [ ! -f OUTCAR ]; then
  echo "[FATAL] OUTCAR not found, stop hook." >> "$LOG_FILE"
  exit 10
fi
tail -n 80 OUTCAR | grep -qiE "total cpu time used|voluntary context switches" || {
  echo "[FATAL] OUTCAR does not show normal termination." >> "$LOG_FILE"
  exit 11
}

# Optional cleanup
if is_lobster_stage; then
  if [ -z "${LOBSTER_BIN}" ] || [ ! -x "${LOBSTER_BIN}" ]; then
    echo "[FATAL] LOBSTER_BIN not set or not executable: ${LOBSTER_BIN}" >> "${LOG_FILE}"
    exit 20
  fi

  # lobsterin 由 VaspInputMaker.write_lobster 写入，这里只做存在性检查
  for f in lobsterin WAVECAR POSCAR POTCAR INCAR KPOINTS; do
    if [ ! -s "$f" ]; then
      echo "[FATAL] LOBSTER stage missing required file: $f" >> "${LOG_FILE}"
      exit 21
    fi
  done

  "${LOBSTER_BIN}" > lobster.log 2>&1

  if [ ! -s lobsterout ]; then
    echo "[FATAL] lobsterout missing/empty" >> "${LOG_FILE}"
    exit 22
  fi
fi

# Optional cleanup
# lobster 阶段不删 WAVECAR
if is_lobster_stage; then
  rm -f REPORT EIGENVAL IBZKPT PCDAT PROCAR XDATCAR FORCECAR || true
else
  rm -f CHG* REPORT EIGENVAL IBZKPT PCDAT PROCAR WAVECAR XDATCAR FORCECAR || true
fi

echo "run complete on $(hostname): $(date) $(pwd)" >> ~/job.log

PYTHON="{{ python_bin }}"
if [ -z "${PYTHON}" ]; then
  PYTHON=/data2/home/luodh/anaconda3/envs/workflow/bin/python
fi

"${PYTHON}" "${WORK_ROOT}/hook.py" --params "${PARAMS_FILE}" mark-done --workdir "${WORKDIR}" >> "$LOG_FILE" 2>&1
