# asm images
# (asm file pixel intensity)

import numpy as np
import os.path
import array
from PIL import Image

# read the asm text file as a binary file
def get_uint8_array(filename):
    f = open(filename, 'rb')
    # hexst = binascii.hexlify(f.read())
    ln = os.path.getsize(filename)
    width = int(ln**0.5)
    rem = ln%width
    a = array.array("B") # unsigned characters
    a.fromfile(f, ln-rem)
    f.close()
    g = np.reshape(a, (int(len(a)/width), int(width)))
    g = np.uint8(g)
    return g

filename = "files/01SuzwMJEIXsK7A8dQbl.asm"
img = Image.fromarray(get_uint8_array(filename))
img.save("01SuzwMJEIXsK7A8dQbl.png")
