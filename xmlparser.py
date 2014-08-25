#!/usr/bin/env python2.7

import argh
import xml.etree.ElementTree as ET


class XML(object):
    """Simple wrapper class for xml.etree.ElementTree.

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
                # params[name] = value

        self.__dict__.update(params)

        self.nyout = self.modes*self.points_per_halfwave
        self.dx = self.W/(self.nyout + 1.)
        self.dy = self.dx
        self.r_nx = int(self.L/self.dx)
        self.r_ny = int(self.W/self.dy)

        params.update(self.__dict__)

        return params


def parse_xml(infile='input.xml'):
    """Return a list of parameters from the given input xml."""
    xml = XML(infile)

    print """
        XML-Settings from {xml}

            modes:  {modes}
            pphw:   {points_per_halfwave:n}
            W:      {W}
            L:      {L:.6f}
            nyout:  {nyout:n}
            dx:     {dx:.6f}
            dy:     {dy:.6f}
            r_nx:   {r_nx}
            r_ny:   {r_ny}
        """.format(**xml.params)


if __name__ == '__main__':
    argh.dispatch_command(parse_xml)
