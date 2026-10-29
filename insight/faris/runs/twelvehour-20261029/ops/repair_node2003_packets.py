from pathlib import Path
import json,subprocess,hashlib
R=Path('/mnt/home/faris/logika-twelvehour-20261029');HOST='faris@gpu-labkc.ub.ac.id';remote=['ssh','-o','BatchMode=yes','-p2003',HOST]
# Only the NEW campaign's symlink is replaced. Old campaign directories stay sealed.
code="from pathlib import Path;r=Path('/mnt/home/faris/logika-twelvehour-20261029');p=r/'packets';assert p.is_symlink();p.unlink();p.mkdir();[(p/f'fold{f}').symlink_to(Path('/mnt/home/faris/logika-sixhour-broad-20261028/packets')/f'fold{f}',target_is_directory=True)for f in [3,4]]"
subprocess.run(remote+['python3 -c '+__import__('shlex').quote(code)],check=True)
for f in [0,1,2]:
 destination=str(R/f'packets/fold{f}');subprocess.run(remote+['mkdir','-p',destination],check=True)
 source=R/f'packets/fold{f}'
 files=[str(source/n)for n in ['MANIFEST.json','features.parquet','roles.npz','inner_roles.npz','anchor.npz']]
 subprocess.run(['rsync','-az','-e','ssh -p2003',*files,HOST+':'+destination+'/'],check=True)
code="""from pathlib import Path
import hashlib,json
r=Path('/mnt/home/faris/logika-twelvehour-20261029');i=json.load(open(r/'INPUTS.json'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for f in range(5):
 p=r/f'packets/fold{f}';m=json.load(open(p/'MANIFEST.json'));assert sha(p/'MANIFEST.json')==i['packet_manifests'][str(f)]
 for n,k in [('features.parquet','feature_sha256'),('roles.npz','roles_sha256'),('inner_roles.npz','inner_roles_sha256'),('anchor.npz','anchor_sha256')]:assert sha(p/n)==m[k]
print('All fivefold node2003 sourcepacket hashes verified; no oldcampaign modified')
"""
subprocess.run(remote+['python3 -c '+__import__('shlex').quote(code)],check=True)
