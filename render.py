#!/usr/bin/env python2.7

import argh
import os
import subprocess
from subprocess import CalledProcessError
import sys


def povray(ppm=None, jpeg=None, scriptfile="scene.pov",
           outfile="out.png", width=4800, height=2250,
           editor=False):
    """Write the .pov script and render scene based on the input .ppm and
    .jpeg files.

        Parameters:
        -----------
            ppm, jpeg: str
                Input files used to render povray scene.
            scriptfile: str
                Output file for the povray source.
            outfile: str
                Output image file.
            width, height: int
                Output image dimensions.
            editor: bool
                Whether to open the source file with the editor defined via the
                $EDITOR variable before rendering.
    """

    if ppm is None or jpeg is None:
        raise Exception("Error: .ppm and .jpeg files are both required!")

    scene = """
        #include "colors.inc"
        #include "textures.inc"

        #declare PPM = "{ppm}"
        #declare JPEG = "{jpeg}"

        #declare zoom=10000000;
        #declare field_height=1;

        #declare ambient_value = 0.7;
        #declare diffuse_value = 0.1;
        #declare transmit_value = 0.0;
        #declare light_height = 10;

        #declare cc=<0,1,0>;

        background {{
            color White
        }}

        camera {{
          orthographic
          location <-2.5*zoom,5*zoom,7*zoom>
          right cc*image_width/image_height
          direction <2.5*zoom,-9*zoom,-7*zoom>
          angle 0.00001
          look_at <0, 0,0>
        }}

        height_field {{
            ppm PPM
            smooth
            water_level -10.0
            texture {{
                pigment {{
                    image_map {{
                        jpeg JPEG
                        transmit all transmit_value
                        }}
                rotate x*90
                }}
                finish {{
                    ambient ambient_value
                    diffuse diffuse_value
                    metallic
                }}
            }}
            scale <14,field_height,-2>
            translate<-7,0,1>
            no_shadow
        }}

        light_source {{
            <0,light_height,0>
            color White
            parallel
            point_at<0,0,0>
            media_interaction 1
            media_attenuation 1
            fade_distance 1
        }}
    """.format(ppm=ppm, jpeg=jpeg)

    # only write scene if no external .pov source is supplied
    if scriptfile == 'scene.pov':
        with open(scriptfile, "w") as f:
            f.write(scene)

    if editor:
        EDITOR = os.environ.get('EDITOR')
        cmd = "{} {}".format(EDITOR, scriptfile)
        try:
            subprocess.check_call(cmd, shell=True)
        except CalledProcessError:
            sys.exit("Aborting...")

    params = { 'W': width,
               'H': height,
               'I': scriptfile,
               'O': outfile
               }
    cmd = "povray -W{W} -H{H} -I{I} -D -O{O}".format(**params)
    subprocess.check_call(cmd.split())


if __name__ == '__main__':
    argh.dispatch_command(povray)
