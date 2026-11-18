"""Network and file sizes shared by the golden model, the vector generator and
(via the generated vectors/config.svh) the RTL testbenches."""
IMG = 48          # input image is IMG x IMG grayscale
K = 3             # conv kernel size
C_OUT = 4         # number of conv filters
POOL = 2          # max-pool window
N_CLASSES = 2     # 0 = not stressed, 1 = stressed
