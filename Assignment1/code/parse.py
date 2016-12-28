import sys

from providedcode.transitionparser import TransitionParser
from providedcode.dependencygraph import DependencyGraph

if __name__ == '__main__':
    tp = TransitionParser.load(sys.argv[1])
    for s in sys.stdin:
        ss = DependencyGraph.from_sentence(s)
        parsed = tp.parse([ss])
        print parsed[0].to_conll(10).encode('utf-8')