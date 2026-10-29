"""Resume the sealed execution loop with atomic NPZ publication environment.

Models, inputs, recipe, RNG and original clock are unchanged. Preserve and count
healthy existing units when establishing capacity; never restart their fits.
"""
from pathlib import Path
import argparse,importlib.util,subprocess,sys,time
R=Path('/mnt/home/faris/logika-twelvehour-20261029')
sys.path.insert(0,str(R/'source'))
spec=importlib.util.spec_from_file_location('sealed_queue',R/'source/queue_jobs.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
original=subprocess.run
def wrapped(args,*a,**kw):
    if args and args[0]=='systemd-run' and any(str(x).startswith('--unit=logika-twelve-')for x in args):
        args=list(args);old=next(x for x in args if str(x).startswith('--unit='));pool=next(p for p in ['cpu2001','cpu2003','gpu2001']if p in old)
        # Existing worker units are independent of the retired dispatcher.
        limit=3 if pool=='cpu2003'else 2
        while True:
            clock=module.check_seal();assert time.time()<clock['admission_end_epoch'] and time.time()<clock['model_deadline_epoch']-20
            units=original(['systemctl','--user','list-units','--state=running','--no-legend','--plain','logika-twelve-'+pool+'-*'],capture_output=True,text=True).stdout
            count=len([line for line in units.splitlines()if '.service'in line])
            if count<limit:break
            time.sleep(3)
        args[args.index(old)]=old+'-atomic3'
        index=next(i for i,x in enumerate(args)if x.startswith('/mnt/home/faris/') and x.endswith('/python'))
        args.insert(index,'--setenv=PYTHONPATH='+str(R/'ops/runtime_hooks'))
    return original(args,*a,**kw)
# External owner designs must never be dispatched on the labs.
original_catalog=module.catalog
module.catalog=lambda:{k:v for k,v in original_catalog().items()if not v.get('resource_owner')}
original_tasks=module.tasks
module.tasks=lambda:{k:v for k,v in original_tasks().items()if v['design']in module.catalog()}
original_dump=module.dump
def deferred_dump(value,path):
    path=Path(path)
    if path.name=='QUEUE_gpu2001_FINAL.json'and(R/'COLAB_RESOURCE_ADDENDUM.json').exists()and not(R/'EXTERNAL_BACKENDS_COLLECTION_COMPLETE.json').exists():
        path=path.with_name('DEFERRED_'+path.name)
    return original_dump(value,path)
module.dump=deferred_dump
module.subprocess.run=wrapped
a=argparse.ArgumentParser();a.add_argument('--pool',required=True);module.main(a.parse_args().pool)
