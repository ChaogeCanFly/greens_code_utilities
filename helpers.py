#!/usr/bin/env python2.7
"""A collection of helper functions for recurring tasks.

    convert_to_complex(s):
        Convert a string of the form (x,y) to a complex number z = x+1j*y.

    replace_in_file(infile, outfile, **replacements):
        Replace some lines in an input file and write to output file.
        The replacements are supplied via an dictionary.
"""
import os
import re


def convert_to_complex(s):
    """Convert a string of the form (x,y) to a complex number z = x+1j*y."""

    regex = re.compile(r'\(([^,\)]+),([^,\)]+)\)')
    x, y = map(float, regex.match(s).groups())
    return x + 1j*y


def natural_sorting(text, args="delta", sep="_"):
    """Sort a text with respect to a given argument value."""

    s = text.split(sep)
    index = lambda text: [ s.index(arg) for arg in args ]
    alphanum_key = lambda text: [ float(s[i+1]) for i in index(text) ]

    return sorted(text, key=alphanum_key)


def replace_in_file(infile, outfile, **replacements):
    """Replace some lines in an input file and write to output file. The
    replacements are supplied via an dictionary."""

    with open(infile) as src_xml:
        src_xml = src_xml.read()

    for src, target in replacements.iteritems():
        src_xml = src_xml.replace(src, target)

    out_xml = os.path.abspath(outfile)
    with open(out_xml, "w") as out_xml:
        out_xml.write(src_xml)
