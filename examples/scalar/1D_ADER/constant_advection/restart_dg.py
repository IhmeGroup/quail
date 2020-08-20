from constant_advection import *

Restart = {
    "File" : "Data_final.pkl",
    "StartFromFileTime" : True
}

TimeStepping.update({
    "FinalTime" : 1.,
    "num_time_steps" : 40,
    "TimeScheme" : "RK4",
})

Output = {
    "Prefix" : "dg",
}

Numerics["Solver"] = "DG"
