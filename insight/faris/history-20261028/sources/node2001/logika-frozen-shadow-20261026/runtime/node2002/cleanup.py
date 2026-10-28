from pathlib import Path
for p in ['/mnt/home/faris/logika-frozen-shadow-20261026/runtime/node2002/credential', '/mnt/home/faris/logika-frozen-shadow-20261026/runtime/node2002/askpass']:
 Path(p).unlink(missing_ok=True)
