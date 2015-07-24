#!/bin/bash

cat eigen.*.log* | grep "Param r_n" | uniq | awk '{$4 = sprintf("%.f", $4); print $0}'
