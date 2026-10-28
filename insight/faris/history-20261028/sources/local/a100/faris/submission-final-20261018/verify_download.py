"""Independent stdlib verification of transferred artifacts and CSV aggregation."""
import array,ast,csv,hashlib,itertools,json,math,struct,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def vector(z,key):
    data=z.read(key+'.npy');assert data[:6]==b'\x93NUMPY';version=data[6]
    if version==1:size=struct.unpack('<H',data[8:10])[0];start=10
    else:size=struct.unpack('<I',data[8:12])[0];start=12
    header=ast.literal_eval(data[start:start+size].decode('latin1').strip());assert not header['fortran_order'] and len(header['shape'])==1
    dtype=header['descr'];types={'f8':'d','f4':'f','i8':'q','i4':'i'};kind=dtype[1:];assert kind in types,dtype;out=array.array(types[kind]);out.frombytes(data[start+size:])
    if dtype[0] in ['<','>'] and (dtype[0]=='<')!=(sys.byteorder=='little'):out.byteswap()
    assert len(out)==header['shape'][0];return out

def ks(labels,values):
    pairs=sorted(zip(values,labels));pos=sum(labels);neg=len(labels)-pos;a=b=0;maximum=0.
    for value,group in itertools.groupby(pairs,key=lambda x:x[0]):
        for _,label in group:a+=label;b+=1-label
        maximum=max(maximum,abs(a/pos-b/neg))
    return maximum

def main():
    complete=json.loads((ROOT/'artifacts/complete.json').read_text());assert complete['state']=='completed';assert complete['verification']['passed']
    for name,h in complete['models'].items():assert sha(ROOT/name)==h,('model',name)
    for name,h in json.loads((ROOT/'research_source_manifest.json').read_text()).items():assert sha(ROOT/'research_source'/name)==h,('source',name)
    for name,h in json.loads((ROOT/'runner_source_manifest.json').read_text()).items():assert sha(ROOT/name)==h,('runner',name)
    plan=json.loads((ROOT/'plan.json').read_text());dataset=ROOT.parents[2]/'dataset'
    for key,name in [('train','train_applications.csv'),('test','test_applications.csv'),('sample','sample_submission(3).csv')]:assert sha(dataset/name)==plan['input_sha256'][key],('input',name)
    with (dataset/'train_applications.csv').open(newline='') as f:train=list(csv.DictReader(f))
    with (dataset/'test_applications.csv').open(newline='') as f:test=list(csv.DictReader(f))
    with (dataset/'sample_submission(3).csv').open(newline='') as f:sample=list(csv.DictReader(f))
    with (ROOT/'artifacts/submission.csv').open(newline='') as f:
        reader=csv.DictReader(f);assert reader.fieldnames==['application_id','default_probability'];submission=list(reader)
    assert len(submission)==len(test)==30000 and len(train)==70000
    ids=[x['application_id'] for x in submission];assert ids==[x['application_id'] for x in sample];assert len(set(ids))==30000
    values=[float(x['default_probability']) for x in submission];assert all(math.isfinite(p) and 0<=p<=1 for p in values)
    positions={row['application_id']:i for i,row in enumerate(test)}
    with zipfile.ZipFile(ROOT/'artifacts/streams.npz') as z:
        rows=vector(z,'rows');assert list(rows)==list(range(70000,100000));arrays=[vector(z,key) for key in plan['selected']['parents']]
    spec=plan['selected'];assert spec['method']=='mean';error=0.
    for appid,p in zip(ids,values):
        index=positions[appid];expected=sum(w*a[index] for w,a in zip(spec['weights'],arrays));error=max(error,abs(p-expected))
    assert error<1e-12,error
    metrics={}
    for label,file in [('development','selected_screen.npz'),('reused_validation','selected_validation.npz')]:
        with zipfile.ZipFile(ROOT/'evidence'/file) as z:row=vector(z,'rows');pred=vector(z,'prediction')
        target=[int(train[i]['target']) for i in row];value=ks(target,pred);expected=complete['development']['screen']['ks'] if label=='development' else complete['reused_validation']['ks'];assert abs(value-expected)<1e-12;metrics[label]=value
    h=sha(ROOT/'artifacts/submission.csv');assert h==complete['submission_sha256']
    result={'passed':True,'verified_model_files':len(complete['models']),'submission_sha256':h,'rows':30000,'exact_sample_order':True,'independent_stream_reaggregation_max_error':error,'independent_tie_aware_ks':metrics,'confirmation_is_selection_data':True,'remote_all_model_replay_rows':complete['verification']['all_rows_replayed'],'no_upload':True}
    (ROOT/'local_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
