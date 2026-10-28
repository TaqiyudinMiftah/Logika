"""Export retained fitted input columns and source dependencies, without fitting."""
from pathlib import Path
import joblib
from benchmark import ROOT,bank,read,write


def main():
    index=read(ROOT/'artifacts/pipeline_index.json');b=bank();items=[]
    for key,item in index['champion'].items():
        bundle=joblib.load(ROOT/item['model']);pre=bundle['preprocessing']
        names=pre.audit['selected_features']
        items.append({'id':key,'original_id':item['component']['original_id'],
            'role':'preserved champion input to trimmed mean','preprocessing':pre.audit,
            'columns':[{'name':n,'lineage':b.manifest['info'].get(n,{'sources':[],'unresolved':True})} for n in names],
            'recipe':item['component']['recipe']})
    c=index['addition'];folder=ROOT/index['addition_folder']
    pre=joblib.load(folder/'pipeline.joblib')['preprocessing'] if c['kind'] in ['historical','resolved'] else joblib.load(folder/'model/preprocessing.joblib')
    names=pre.audit['selected_features']
    items.append({'id':c['id'],'original_id':c['original_id'],'role':'small probability addition','weight':index['addition_weight'],
        'preprocessing':pre.audit,'columns':[{'name':n,'lineage':b.manifest['info'].get(n,{'sources':[],'unresolved':True})} for n in names],
        'configuration':c})
    write(ROOT/'artifacts/feature_lineage.json',{'streams':items,
        'reduction_note':'Column lists describe selected original inputs. Appended fitted projection components are represented by their saved transformers and output_features, not mislabelled as original columns.',
        'all_fitted_transformers_retained':True})
    print('Exported lineage for',len(items),'fitted model streams')


if __name__=='__main__':main()
