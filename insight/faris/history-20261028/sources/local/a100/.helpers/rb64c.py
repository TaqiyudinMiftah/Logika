import sys, json, ssl, uuid, datetime, base64
import websocket
KID='0741e853-d6eb-4aed-acc7-25d728f2effc'
tok=open('sysinfo_fariselhakim.txt').read().strip().split('=',1)[1]
raw = base64.b64decode(sys.argv[1]).decode()
timeout_s = int(sys.argv[2]) if len(sys.argv)>2 else 60
ws_url='wss://node480-ai-hub.ub.ac.id/user/fariselhakim/api/kernels/'+KID+'/channels?session_id='+uuid.uuid4().hex
ws = websocket.create_connection(ws_url, header=['Authorization: token '+tok], origin='https://node480-ai-hub.ub.ac.id', sslopt={'cert_reqs': ssl.CERT_REQUIRED}, timeout=25)
sid=uuid.uuid4().hex; mid=uuid.uuid4().hex
inner = base64.b64encode(raw.encode()).decode()
code = "import subprocess,base64\n_p = subprocess.run(base64.b64decode('"+inner+"').decode(), shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)\nprint(_p.stdout, end='')\nprint('[remote exit code: %d]' % _p.returncode)"
msg={'header':{'msg_id':mid,'username':'codex','session':sid,'date':datetime.datetime.now(datetime.timezone.utc).isoformat(),'msg_type':'execute_request','version':'5.3'},'parent_header':{},'metadata':{},'content':{'code':code,'silent':False,'store_history':False,'user_expressions':{},'allow_stdin':False,'stop_on_error':True},'channel':'shell','buffers':[]}
ws.send(json.dumps(msg))
ws.settimeout(timeout_s)
out=[]
while True:
  r=json.loads(ws.recv())
  if r.get('parent_header',{}).get('msg_id')!=mid: continue
  mt=r.get('msg_type') or r.get('header',{}).get('msg_type')
  c=r.get('content',{});
  if mt=='stream': out.append(c.get('text',''))
  elif mt=='status' and c.get('execution_state')=='idle': break
  elif mt=='error': out.append('\nWS-ERROR: '+str(c)[:500]); break
ws.close()
print(''.join(out))
