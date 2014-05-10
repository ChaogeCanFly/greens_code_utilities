#!/usr/bin/env python

import numpy as np
from math import *
import re, os, glob
import argparse as ap

#------------------------------
#	Parse command-line
#------------------------------
parser = ap.ArgumentParser()
parser.add_argument("-i", "--input-xml", default="input.xml",
                    type=str, help="Input file for executable")
args = parser.parse_args()

#------------------------------
#	Functions	
#------------------------------

def getInitValues(xml):
	"""
	Parse kMin, kMax, kN from input xml file.
	"""
	prefix = "kFp_"
	regex = re.compile(r'([0-9][0-9]*)+(\.[0-9][0-9]?)?')
	kValues = []

	for line in xml:
		if (prefix in line) and ("receiver" not in line):
				digit = regex.findall(line)
				kValues.append(digit[0][0]+digit[0][1])
	kValues = [ kValues[i] for i in (0,1,2) ]

	return map(float, kValues)

def getWireWidth(xml):
	"""
	Parse width of the waveguide from input xml file.
	"""
	var = 'name="W"'

	regex = re.compile(r'([0-9][0-9]*)+(\.[0-9][0-9]?)?')

	for line in xml:
		if var in line:
			wireWidth = regex.findall(line)
			return float(wireWidth[0][0])
			

def getTransmission():
	"""
	Calculate transmission as a function of iterations of greens_code.
	"""
	xml = open(args.input_xml, "r")
	kMin, kMax, kN = getInitValues(xml)
	xml.seek(0, 0)	# rewind file
	wireWidth = getWireWidth(xml)
	xml.close()
	dk = (kMax-kMin)/kN * pi/wireWidth

	print
	print "kMin = ", kMin
	print "kMax = ", kMax
	print "kN = ", kN
	print "d = ", wireWidth
	print

	for file in glob.glob("*.cmplx"):
		print "Processing file ", file
		i, j =  map(int,filter(str.isdigit, file))
		cmplx = open(file, "r")
		trans = []

		for line in cmplx:
			a = re.split(r'[ ,()]',line)
			a = [float(x) for x in a if (x != "\n") and (x != "")]
			trans.append([kMin*pi/wireWidth + a[0]*dk, a[1]**2+a[2]**2])
		if file.startswith("trans"):
			n = "T"
		elif file.startswith("refl"):
			n = "R"

		f = open("%s%i%i.dat" % (n,i-1,j-1),"w")
		for (k,T) in trans:
			f.write("%.10f \t %.16f \n" % (k,T) )
		f.close()

#------------------------------
# Main	
#------------------------------
getTransmission()

