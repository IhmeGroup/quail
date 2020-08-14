from p0 import *

Restart = {
    "File" : "p0_final.pkl",
    "StartFromFileTime" : True
}

TimeStepping.update({
    "EndTime" : 48.,
    "num_time_steps" : 1000,
})

Numerics["InterpOrder"] = 1
Numerics["InterpBasis"] = "LegendreQuad"
Output["Prefix"] = "p1"
