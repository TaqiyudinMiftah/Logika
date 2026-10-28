import subprocess,sys,json,time
from pathlib import Path
r=Path('/home/faris/logika-model-arena-20261025')
key='EXAONE-Tabular'
cmd=['/home/faris/logika-model-arena-20261025/env311/bin/python', '-m', 'pip', 'install', '--no-cache-dir', '--no-deps', '--target', '/home/faris/logika-model-arena-20261025/storage/extensions', 'exaonetabular @ git+https://github.com/LGAI-Research/EXAONE-Tabular.git@8638e07d09fad154249bd75ba4786181491f7025']
start=time.time()
with (r/'evidence'/('install_git_'+key+'.log')).open('x') as out:
    p=subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT,timeout=900)
print(json.dumps({'key':key,'returncode':p.returncode,'seconds':time.time()-start}),flush=True)
key='FTFM'
cmd=['/home/faris/logika-model-arena-20261025/env311/bin/python', '-m', 'pip', 'install', '--no-cache-dir', '--no-deps', '--target', '/home/faris/logika-model-arena-20261025/storage/extensions', 'ftfm @ git+https://github.com/machinelearningnuremberg/ftfm-official.git@7e5e237daa26ef20f0faec1650d000b04de5122e']
start=time.time()
with (r/'evidence'/('install_git_'+key+'.log')).open('x') as out:
    p=subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT,timeout=900)
print(json.dumps({'key':key,'returncode':p.returncode,'seconds':time.time()-start}),flush=True)
key='LimiX-2'
cmd=['/home/faris/logika-model-arena-20261025/env311/bin/python', '-m', 'pip', 'install', '--no-cache-dir', '--no-deps', '--target', '/home/faris/logika-model-arena-20261025/storage/extensions', 'LimiX @ git+https://github.com/limix-ldm-ai/LimiX.git@774aa3e1a994cbe38f33758e3d663e9951855554']
start=time.time()
with (r/'evidence'/('install_git_'+key+'.log')).open('x') as out:
    p=subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT,timeout=900)
print(json.dumps({'key':key,'returncode':p.returncode,'seconds':time.time()-start}),flush=True)
key='SAP-RPT-OSS'
cmd=['/home/faris/logika-model-arena-20261025/env311/bin/python', '-m', 'pip', 'install', '--no-cache-dir', '--no-deps', '--target', '/home/faris/logika-model-arena-20261025/storage/extensions', 'sap_rpt_oss @ git+https://github.com/SAP-samples/sap-rpt-1-oss.git@a323a0aff976fda4ac43c3196a92406de7689aaa']
start=time.time()
with (r/'evidence'/('install_git_'+key+'.log')).open('x') as out:
    p=subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT,timeout=900)
print(json.dumps({'key':key,'returncode':p.returncode,'seconds':time.time()-start}),flush=True)
key='TabFM'
cmd=['/home/faris/logika-model-arena-20261025/env311/bin/python', '-m', 'pip', 'install', '--no-cache-dir', '--no-deps', '--target', '/home/faris/logika-model-arena-20261025/storage/extensions', 'tabfm[pytorch] @ git+https://github.com/google-research/tabfm.git@fbb665569425fd2f490c6576b3af967876fe11ff']
start=time.time()
with (r/'evidence'/('install_git_'+key+'.log')).open('x') as out:
    p=subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT,timeout=900)
print(json.dumps({'key':key,'returncode':p.returncode,'seconds':time.time()-start}),flush=True)
key='Xiaomi-TabLDM'
cmd=['/home/faris/logika-model-arena-20261025/env311/bin/python', '-m', 'pip', 'install', '--no-cache-dir', '--no-deps', '--target', '/home/faris/logika-model-arena-20261025/storage/extensions', 'Xiaomi-TabLDM @ git+https://github.com/xiaomi-research/xiaomi-tabldm.git@d3fc3b7a0644631c8e3c18fa50629440e334f3f4']
start=time.time()
with (r/'evidence'/('install_git_'+key+'.log')).open('x') as out:
    p=subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT,timeout=900)
print(json.dumps({'key':key,'returncode':p.returncode,'seconds':time.time()-start}),flush=True)
