from constant_advection import *

Restart = {
    "File" : "Data_final.pkl",
    "StartFromFileTime" : True
}

TimeStepping.update({
    "FinalTime" : 1.,
    "num_time_steps" : 40,
    "TimeScheme" : "ADER",
})

Output = {
    "Prefix" : "ader",
}

Numerics["Solver"] = "ADERDG"
