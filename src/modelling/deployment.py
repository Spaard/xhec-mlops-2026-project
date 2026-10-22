"""Deployment of the training flow, which retrains the model every week.

Run this script while the Prefect server is running: it registers the deployment
with the server, then keeps running to execute the scheduled training runs. A run
can also be started at any time from the Prefect UI.
"""

from modelling.main import main

DEPLOYMENT_NAME = "weekly-retraining"
RETRAINING_SCHEDULE = "0 3 * * 1"  # every Monday at 03:00 UTC

if __name__ == "__main__":
    main.serve(name=DEPLOYMENT_NAME, cron=RETRAINING_SCHEDULE)
