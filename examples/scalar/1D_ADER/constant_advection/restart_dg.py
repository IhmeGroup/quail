from constant_advection import *

Restart = {
    "File" : "Data_final.pkl",
    "StartFromFileTime" : True
}

TimeStepping.update({
    "EndTime" : 1.,
    "NumTimeSteps" : 40,
    "TimeScheme" : "RK4",
})

Output = {
    "Prefix" : "dg",
}

Numerics["Solver"] = "DG"
