import general
import numerics.limiting.positivitypreserving as pp_limiter

def set_limiter(limiter_type, physics_type):
	'''
    Method: set_limiter
    ----------------------------
	selects limiter bases on input deck

    INPUTS:
		limiterType: type of limiter selected (Default: None)
	'''
	if limiter_type is None:
		return None
	elif general.LimiterType[limiter_type] is general.LimiterType.PositivityPreserving:
		limiter_ref = pp_limiter.PositivityPreserving
	elif general.LimiterType[limiter_type] is general.LimiterType.ScalarPositivityPreserving:
		limiter_ref = pp_limiter.ScalarPositivityPreserving
	elif general.LimiterType[limiter_type] is general.LimiterType.PositivityPreservingChem:
		limiter_ref = pp_limiter.PositivityPreservingChem
	else:
		raise NotImplementedError

	limiter = limiter_ref(physics_type)

	return limiter