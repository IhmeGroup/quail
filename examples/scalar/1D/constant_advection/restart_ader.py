from constant_advection import *

Restart = {
    "File" : "Data_final.pkl",
    "StartFromFileTime" : True
}

TimeStepping.update({
    "FinalTime" : 1.,
    "NumTimeSteps" : 40,
    "TimeStepper" : "ADER",
})

Output = {
    "Prefix" : "ader",
}

Numerics["Solver"] = "ADERDG"
