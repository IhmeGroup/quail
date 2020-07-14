from constant_advection import *

Restart = {
    "File" : "Data_final.pkl",
    "StartFromFileTime" : True
}

TimeStepping.update({
    "EndTime" : 1.,
    "NumTimeSteps" : 40,
    "TimeScheme" : "ADER",
})

Output = {
    "Prefix" : "ader",
}

Numerics["Solver"] = "ADERDG"
