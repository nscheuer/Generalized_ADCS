from ADCS.CONOPS.rulebook import Rule, Rulebook
from ADCS.CONOPS.rules import is_tumbling
from ADCS.ADCS import ADCS

def test_pointing_conops() -> None:
    attitude_estimator_rules = [Rule(True, "SRUAKF")]
    orbital_estimator_rules = [Rule(True, "GPS")]
    controller_rules = [Rule(is_tumbling, "DETUMBLE"), Rule(True, "POINT")]

    rulebook = Rulebook(attitude_estimator_rules, orbital_estimator_rules, controller_rules)

    orbit_estimators = {"GPS": }