import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
from metaexp.common import read,sha
from design_extension import Resolver
from research_worker import install_guard


class ExtensionTests(unittest.TestCase):
    def test_no_new_recipe_matches_previous_effective_representation(self):
        root=Path(__file__).resolve().parent;catalogue=read(root/'catalogue.json');cfg=read(root/'campaign.json')
        parent=root.with_name(cfg['parent_campaign']);resolver=Resolver(read(root/'bank/manifest.json'))
        old=set()
        for rep in read(parent/'catalogue.json')['representations']:
            try:old.add(resolver.signature(rep['family'],rep['config'])[0])
            except ValueError:pass
        new=[r for r in catalogue['representations'] if r.get('wave')==2]
        actual=[resolver.signature(r['family'],r['config'])[0] for r in new]
        self.assertEqual(len(actual),len(set(actual)));self.assertFalse(set(actual)&old)
        self.assertEqual(catalogue['new_pipeline_recipes'],4*len(new))

    def test_fitting_and_feature_code_unchanged(self):
        root=Path(__file__).resolve().parent;cfg=read(root/'campaign.json');parent=root.with_name(cfg['parent_campaign'])
        for name in ['pipeline.py','features.py','worker.py','common.py','ensembles.py']:
            self.assertEqual(sha(root/'metaexp'/name),sha(parent/'metaexp'/name))

    def test_clock_is_parent_clock(self):
        root=Path(__file__).resolve().parent;cfg=read(root/'campaign.json');parent=read(root.with_name(cfg['parent_campaign'])/'campaign.json')
        self.assertEqual(cfg['started'],parent['started']);self.assertEqual(cfg['deadline'],parent['deadline'])
        self.assertTrue(cfg['research_only'])

    def test_confirmation_reads_are_rejected(self):
        original=np.load
        try:
            install_guard()
            for name in ['confirmation_y.npy','confirmation_rows.npy']:
                with self.assertRaises(PermissionError):np.load('/not/opened/'+name)
            with tempfile.TemporaryDirectory() as d:
                path=Path(d)/'development_y.npy';np.save(path,[0,1]);self.assertEqual(np.load(path).tolist(),[0,1])
        finally:np.load=original

if __name__=='__main__':unittest.main(verbosity=2)
