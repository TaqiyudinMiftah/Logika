"""Run sealed composition, resolving only explicitly approved retry folders."""
from pathlib import Path
import inspect,sys
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'source'))
import compose
from common import check_seal,load,sha,dump,utc
check_seal()
source=inspect.getsource(compose.completed_raw)
old="    root = R / 'runs' / task_id / 'attempt1'"
new="""    attempt = int(load('APPROVED_RETRIES.json').get(task_id,1)) if (R/'APPROVED_RETRIES.json').exists() else 1
    assert attempt in {1,2}
    roots = list((R/'runs'/task_id).glob('attempt*/RESULT.json'))
    assert len(roots) <= 1, 'Multiple successful attempts cannot be silently selected'
    root = R / 'runs' / task_id / f'attempt{attempt}'"""
assert source.count(old)==1;source=source.replace(old,new)
exec(compile(source,str(Path(__file__)),'exec'),compose.__dict__)
compose.main()
dump(dict(UTC=utc(),source_sha256=sha(__file__),sealed_composition_sha256=sha(R/'source/compose.py'),change='Resolve only explicit approved attempt1/2 folder; reject multiple successful attempts. Every original prediction/source/label-boundary predicate and numerical composition remains unchanged.',scientific_source_or_selection_changed=False),R/'COMPOSITION_EXECUTION_ADAPTER.json')
