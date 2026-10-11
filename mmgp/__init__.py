# Copyright (c) 2024-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import enum    
class profile_type(int, enum.Enum): 
    @staticmethod
    def tostr(v):
        if v == 1:
            s= "HighRAM_HighVRAM" 
        elif v == 2:
            s ="HighRAM_LowVRAM"
        elif v == 3: 
            s = "LowRAM_HighVRAM"
        elif v == 4:
            s = "LowRAM_LowVRAM"
        else:
            s = "VerylowRAM_LowVRAM"
        return s
    
    HighRAM_HighVRAM  = 1
    HighRAM_LowVRAM  = 2
    LowRAM_HighVRAM  = 3
    LowRAM_LowVRAM  = 4
    VerylowRAM_LowVRAM  = 5

