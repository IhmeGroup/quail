from p0 import *

Restart = {
	"File" : "p0_final.pkl",
	"StartFromFileTime" : True
}

TimeStepping.update({
	"FinalTime" : 48.,
	"NumTimeSteps" : 1000,
})

Numerics["SolutionOrder"] = 1
Numerics["SolutionBasis"] = "LegendreQuad"
Output["Prefix"] = "p1"
