#!/usr/bin/env python2.7

import argh
try:
    from lxml import etree as ET
except ImportError:
    import xml.etree.ElementTree as ET


class XML(object):
    """Simple wrapper class for xml.etree.ElementTree/lxml.etree.

        Parameters:
        -----------
            xml: str
                Input xml file.

        Attributes:
        -----------
            root: Element object
            params: dict
                Dictionary of the parameters parsed from the input xml.
    """

    def __init__(self, xml):
        self.xml = xml
        self.root = ET.parse(xml).getroot()
        self.params = self._read_xml()

    def _read_xml(self):
        """Read all variables from the input.xml file."""

        params = {}
        for elem in self.root.iter(tag='param'):
            name = elem.attrib.get('name')
            value = elem.text
            try:
                params[name] = float(value)
            except ValueError:
                pass

        # avoid confusion between parameter 'N' from the input-vector object
        # and the variable 'N' used for modes in the Waveguide class
        try:
            params.pop("N")
            params.pop("v[i]")
        except:
            print "Warning: could not remove 'N' and 'v[i]' parameters."

        nyout = params.get("modes")*params.get("points_per_halfwave")
        dx = params.get("W")/(nyout + 1.)
        dy = dx
        r_nx = int(params.get("L")/dx)
        try:
            r_nx_fixed = params.get("r_nx")
            print "r_nx_fixed", r_nx_fixed
        except:
            pass

        r_ny = int(params.get("W")/dy)

        values = {
                'nyout': nyout,
                'dx': dx,
                'dy': dy,
                'r_nx': r_nx,
                'r_nx_fixed': r_nx_fixed,
                'r_ny': r_ny}

        params.update(values)

        return params


def parse_xml(infile='input.xml'):
    """Return a list of parameters from the given input xml."""
    xml = XML(infile)

    print """
        XML-Settings from {0}

            modes:  {modes}
            pphw:   {points_per_halfwave:n}
            W:      {W}
            L:      {L:.6f}
            nyout:  {nyout:n}
            dx:     {dx:.6f}
            dy:     {dy:.6f}
            r_nx:   {r_nx}
            r_ny:   {r_ny}
        """.format(infile, **xml.params)


if __name__ == '__main__':
    argh.dispatch_command(parse_xml)
