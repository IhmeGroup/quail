from p1 import *

Restart = {
    "File" : "p1_final.pkl",
    "StartFromFileTime" : True
}

TimeStepping.update({
    "FinalTime" : 54.,
    "NumTimeSteps" : 1500,
})

Numerics["SolutionOrder"] = 2
Output["Prefix"] = "p2"
Output["AutoPostProcess"] = False
