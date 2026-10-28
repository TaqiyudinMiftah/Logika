from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ.setdefault(n,'1')

def main():
    parser=argparse.ArgumentParser(description='Unattended feature experiments; frozen 3-fold CV, no agents or competition submission API.')
    parser.add_argument('command',choices=['audit','design','preflight','deploy','launch','status','collect','report','resume','stop','prepare','daemon','coordinate','rpc'])
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);parser.add_argument('--workspace',type=Path)
    parser.add_argument('--host',choices=['pc1','pc2','pc3']);parser.add_argument('--exercise',action='store_true')
    parser.add_argument('--raw-dir',type=Path);parser.add_argument('--cached-dir',type=Path);parser.add_argument('--representations',type=int,default=1024)
    args=parser.parse_args();root=args.root.resolve()
    if args.command=='audit':
        from .reporting import audit
        x=audit(root,args.workspace);print(json.dumps({'historical_records':len(x['records'])}))
    elif args.command=='design':
        from .design import design
        x=design(root,args.representations);print(json.dumps({'registered_pipelines':len(x)}))
    elif args.command=='prepare':
        from .features import prepare_bank
        prepare_bank(root,args.raw_dir,args.cached_dir)
    elif args.command=='preflight':
        from .preflight import run
        run(root,args.host,args.exercise)
    elif args.command in ['launch','resume']:
        from .coordinator import launch
        print(json.dumps(launch(root)))
    elif args.command=='coordinate':
        from .coordinator import Coordinator
        Coordinator(root).run()
    elif args.command=='daemon':
        from .runtime import daemon
        daemon(root,args.host)
    elif args.command=='rpc':
        from .runtime import rpc
        rpc(root)
    elif args.command=='report':
        from .reporting import report
        print(json.dumps(report(root,final=True)))
    elif args.command in ['status','collect','stop']:
        from .common import read
        if args.command=='status':print(json.dumps(read(root/'campaign_status.json'),indent=2));return
        from .coordinator import Coordinator
        from .features import Bank
        c=Coordinator(root);c.bank=Bank(root)
        if args.command=='collect':c.sync()
        else:
            for host in ['pc1','pc2','pc3']:c.tx.call(host,{'action':'stop'})
            subprocess.run(['systemctl','--user','stop','metaexp-'+c.cfg['id']+'-coordinator'],check=True)
    elif args.command=='deploy':
        from .deployment import deploy
        deploy(root,args.workspace)

if __name__=='__main__':main()
