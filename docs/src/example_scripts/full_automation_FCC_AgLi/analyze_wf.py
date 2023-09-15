from jobflow import SETTINGS
from pydantic import parse_obj_as

from WFacer.schema import CeOutputsDocument

store = SETTINGS.JOB_STORE
store.connect()

# Just a random example. You have to check what is your maximum iteration on your own.
max_iter = 10
# Find the output of a trigger job, which should be the CeOutputDocument of the final
# iteration.
job_return = store.query_one({"name": f"agli_fcc_ce_iter_{max_iter}_trigger"})
raw_doc = job_return["output"]
# De-serialize everything.
doc = parse_obj_as(CeOutputsDocument, raw_doc)

# See WFacer.schema for more.
print("Cluster subspace:", doc.cluster_subspace)
print("Wrangler:", doc.data_wrangler)
print("coefficients:", doc.coefs_history[-1])
