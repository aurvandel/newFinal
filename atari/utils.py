import os
import numpy as np
from PIL import Image

IMG_SIZE = None


# Functions
def preprocess_observation(obs):
    global IMG_SIZE
    # Convert to gray-scale and resize it
    image = Image.fromarray(obs, 'RGB').convert('L').resize(IMG_SIZE)
    # Convert image to array and return it
    return np.asarray(image.getdata(), dtype=np.uint8).reshape(image.size[1],
                                                               image.size[0])


def get_next_state(current, obs):
    # Next state is composed by the last 3 images of the previous state and the
    # new observation
    return np.append(current[1:], [obs], axis=0)

def get_CPU_pct():
    CPU_Pct=str(round(float(os.popen('''grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage }' ''').readline()),2))
    return CPU_Pct

def get_mem():
    mem = str(os.popen('free -t -m').readlines())
    """
    Get a whole line of memory output, it will be something like below
    ['             total       used       free     shared    buffers     cached\n', 
    'Mem:           925        591        334         14         30        355\n', 
    '-/+ buffers/cache:        205        719\n', 
    'Swap:           99          0         99\n', 
    'Total:        1025        591        434\n']
     So, we need total memory, usage and free memory.
     We should find the index of capital T which is unique at this string
    """
    T_ind = mem.index('T')
    """
    Than, we can recreate the string with this information. After T we have,
    "Total:        " which has 14 characters, so we can start from index of T +14
    and last 4 characters are also not necessary.
    We can create a new sub-string using this information
    """
    mem_G = mem[T_ind + 14:-4]
    """
    The result will be like
    1025        603        422
    we need to find first index of the first space, and we can start our substring
    from from 0 to this index number, this will give us the string of total memory
    """
    S1_ind = mem_G.index(' ')
    mem_T = mem_G[0:S1_ind]
    """
    Similarly we will create a new sub-string, which will start at the second value. 
    The resulting string will be like
    603        422
    Again, we should find the index of first space and than the 
    take the Used Memory and Free memory.
    """
    mem_G1 = mem_G[S1_ind + 8:]
    S2_ind = mem_G1.index(' ')
    mem_U = mem_G1[0:S2_ind]

    mem_F = mem_G1[S2_ind + 8:]
    return mem_G, mem_T, mem_U, mem_F
