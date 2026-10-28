"""Verify every imported source mapping, bundle/member hash and coverage receipt."""
import argparse
import collections
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import tarfile


def digest(path):
    h=hashlib.sha256()
    with path.open('rb')as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def verify(repo):
    history=repo/'insight/faris/history-20261028'
    index=json.loads((history/'SOURCE_INVENTORY_INDEX.json').read_text())
    status=json.loads((history/'IMPORT_STATUS.json').read_text())
    bundles=json.loads((history/'BUNDLE_INDEX.json').read_text())['bundles']
    expected=collections.defaultdict(dict);files={};counts=collections.Counter();sources=collections.defaultdict(collections.Counter)
    records=0;seen=set()
    for part in index['parts']:
        path=history/part['file'];assert digest(path)==part['sha256'],path
        rows=0
        with gzip.open(path,'rt')as stream:
            for line in stream:
                r=json.loads(line);rows+=1;records+=1
                key=(r['source'],r['path']);assert key not in seen,key;seen.add(key)
                sources[r['source']]['source_records']+=1
                if r['kind']=='file':
                    counts[r['publication']]+=1;sources[r['source']][r['publication']]+=1
                    assert r['publication'] not in {'missing_fetch','source_changed_after_inventory'},r
                if r.get('publication')in{'archived','deduplicated_alias'}:
                    if 'public_archive'in r:
                        old=expected[r['public_archive']].setdefault(r['archive_member'],r['public_sha256']);assert old==r['public_sha256']
                    else:
                        old=files.setdefault(r['public_path'],r['public_sha256']);assert old==r['public_sha256']
                elif r['kind']in{'scan_error','missing_root'}:raise AssertionError(r)
        assert rows==part['rows'],(path,rows)
    assert records==index['rows']==status['source_records']
    assert dict(counts)==status['counts'],(counts,status['counts'])
    for path,sha in files.items():assert digest(repo/path)==sha,path
    checked_members=0
    for b in bundles:
        path=repo/b['public_archive'];assert digest(path)==b['sha256'],path
        members=expected.pop(b['public_archive'])
        found=0
        with tarfile.open(path,'r|gz')as tar:
            for member in tar:
                assert member.isfile() and not Path(member.name).is_absolute() and '..'not in Path(member.name).parts,member.name
                data=tar.extractfile(member).read();sha=hashlib.sha256(data).hexdigest()
                assert members.pop(member.name)==sha,(path,member.name)
                found+=1
        assert not members and found==b['members'],path
        checked_members+=found
    assert not expected
    result={'verified_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'status':'VERIFIED_COMPLETE_FOR_DISCOVERED_ACCESSIBLE_SOURCES',
            'source_records':records,'per_source':{k:dict(v)for k,v in sources.items()},
            'publication_counts':dict(counts),'verified_direct_files':len(files),
            'verified_bundles':len(bundles),'verified_bundle_members':checked_members,
            'all_public_content_hashes_verified':True,'missing_or_changed_public_sources':0,
            'limits':['Coverage is bounded by documented discovered source roots; retired node2002 is not accessed.',
                      'Environments, caches, unrelated competitions, raw data, fitted weights and case-level evidence remain private.',
                      'Source file counts, mirrored aliases, fit attempts and independent experiments are different quantities.',
                      'Hash verification does not validate scientific claims in historical reports.']}
    (history/'ARCHIVE_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
    status['status']='hash_verified_pending_secret_scan_and_remote_push';status['verification']='insight/faris/history-20261028/ARCHIVE_VERIFICATION.json'
    (history/'IMPORT_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);a=p.parse_args();verify(Path(a.repo))
