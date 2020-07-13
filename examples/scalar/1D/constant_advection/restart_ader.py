from LinearAdvection import *

Restart = {
    "File" : "Data_final.pkl",
    "StartFromFileTime" : True
}

TimeStepping.update({
    "EndTime" : 1.,
    "nTimeStep" : 40,
    "TimeScheme" : "ADER",
})

Output = {
    "Prefix" : "ader",
}

Numerics["Solver"] = "ADERDG"
