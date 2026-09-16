import logging

SIM_LEVEL = 15
RESULTS_LEVEL = 17

logging.addLevelName(SIM_LEVEL, "SIM")
logging.addLevelName(RESULTS_LEVEL, "RESULTS")

# 2) Methode definieren
def _log_sim(self, message, *args, **kwargs):
    if self.isEnabledFor(SIM_LEVEL):
        self._log(SIM_LEVEL, message, args, **kwargs)

def _log_results(self, message, *args, **kwargs):
    if self.isEnabledFor(RESULTS_LEVEL):
        self._log(RESULTS_LEVEL, message, args, **kwargs)

# 3) Monkey-patch der Logger-Klasse
logging.Logger.sim = _log_sim
logging.Logger.results = _log_results

# 4) Optional: Root-Logger konfigurieren
logging.basicConfig(
    level=RESULTS_LEVEL,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
