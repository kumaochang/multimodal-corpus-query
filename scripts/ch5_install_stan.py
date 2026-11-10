"""Install a pinned CmdStan into the task's ignored runtime, no global PATH edits."""
import os
import sys
import json
import platform
import subprocess
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUNTIME=ROOT/'data/ch5_stan'
TOOLCHAIN=Path('C:/rtools45/x86_64-w64-mingw32.static.posix/bin')
MSYS=Path('C:/rtools45/usr/bin')
os.environ['PATH']=str(TOOLCHAIN)+os.pathsep+str(MSYS)+os.pathsep+os.environ['PATH']
os.environ['MAKE']=str(MSYS/'make.exe')
for v in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','STAN_NUM_THREADS']:os.environ[v]='1'
import psutil
import cmdstanpy

if __name__=='__main__':
    RUNTIME.mkdir(exist_ok=True)
    target=RUNTIME/'cmdstan-2.37.0'
    started=time.time()
    installing=not (target/'bin/stansummary.exe').exists()
    if installing:
        ok=cmdstanpy.install_cmdstan(version='2.37.0',dir=str(RUNTIME),cores=4,verbose=True,progress=False)
        assert ok,'CmdStan installation failed'
    cmdstanpy.set_cmdstan_path(str(target))
    record=dict(python=sys.version,executable=sys.executable,cmdstanpy=cmdstanpy.__version__,
        cmdstan=str(cmdstanpy.cmdstan_version()),cmdstan_path=str(target),
        compiler=subprocess.check_output([str(TOOLCHAIN/'g++.exe'),'--version'],text=True).splitlines()[0],
        make=subprocess.check_output([str(MSYS/'make.exe'),'--version'],text=True).splitlines()[0],
        machine=platform.platform(),cpu=os.environ.get('PROCESSOR_IDENTIFIER'),logical_cpus=os.cpu_count(),
        physical_cpus=psutil.cpu_count(logical=False),RAM_bytes=psutil.virtual_memory().total,
        disk_free_bytes=psutil.disk_usage(str(ROOT)).free,
        operation='installation' if installing else 'verify_existing_installation',
        operation_seconds=time.time()-started,global_PATH_changed=False,ranking_environment_changed=False)
    (RUNTIME/'environment.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    subprocess.run([sys.executable,'-m','pip','freeze'],stdout=(RUNTIME/'requirements.lock.txt').open('w'),check=True)
    print(json.dumps(record),flush=True)
