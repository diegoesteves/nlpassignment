# -*- coding: utf-8 -*-


class Transition(object):
    """
    This class defines a set of transitions which are applied to a
    configuration to get the next configuration.
    """
    # Define set of transitions
    LEFT_ARC = 'LEFTARC'
    RIGHT_ARC = 'RIGHTARC'
    SHIFT = 'SHIFT'
    REDUCE = 'REDUCE'

    def __init__(self):
        raise ValueError('Do not construct this object!')

    @staticmethod
    def left_arc(conf, relation):
        """Add the arc (b, L, s) to A (arcs), and pop Σ (a stack).

        That is, draw an arc between the next node on the buffer and the next node on the stack, with the label L.

            :param conf: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        if not conf.buffer or not conf.stack:
            return -1
        x = conf.stack[-1]
        if x == 0 or (any(x == c for a, b, c in conf.arcs)):
            return -1
        conf.stack.pop()
        conf.arcs.append((conf.buffer[0], relation, x))

    @staticmethod
    def right_arc(conf, relation):
        """Add the arc (s, L, b) to A (arcs), and push b onto Σ.

            :param conf: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        if not conf.buffer or not conf.stack:
            return -1

        x = conf.stack[-1]
        y = conf.buffer.pop(0)
        conf.stack.append(y)
        conf.arcs.append((x, relation, y))

    @staticmethod
    def reduce(conf):
        """Pop Σ (a stack).

            :param conf: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        if not conf.stack:
            return -1
        x = conf.stack[-1]

        if any(x == c for a, b, c in conf.arcs):
            conf.stack.pop()
        else:
            return -1

    @staticmethod
    def shift(conf):
        """Remove b from B (buffer) and add it to Σ (a stack).

            :param conf: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        if not conf.buffer:
            return -1

        conf.stack.append(conf.buffer.pop(0))
