#!/usr/bin/env python
import re
import sys
from subprocess import getoutput

term = sys.argv[1]
command = f"""grep -r '{term}' ./ | grep -v checkpoints"""
res = getoutput(command).split('\n')
c = re.compile("N20.*ipynb")
a = [c.findall(x) for x in res]
a = sorted({x[0] for x in a if len(a)})[::-1]
for x in a:
    print(x)