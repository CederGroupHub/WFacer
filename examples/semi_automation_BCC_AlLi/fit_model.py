"""Parse and fit cluster expansion."""
import json

from atomate2.vasp.schemas.task import TaskDocument
from jobflow import SETTINGS
from monty.json import jsanitize
from monty.serialization import loadfn
from pydantic import parse_file_as, parse_obj_as

from WFacer.jobs import fit_calculations, parse_calculations, update_document
from WFacer.schema import CeOutputsDocument


# Execute this once all queue tasks has been completed and no job is lost.
def __main__():
    document = parse_file_as(CeOutputsDocument, "document.json")

    iter_id = document.last_iter_id + 1
    project_name = document.project_name

    enum_output = loadfn(f"enum_iter_{iter_id}.json")

    print("Loading TaskDocuments!")
    store = SETTINGS.JOB_STORE
    store.connect()

    new_structures = enum_output["new_structures"]
    if document.enumerated_structures is None:
        struct_id = 0
    else:
        struct_id = len(document.enumerated_structures)
    taskdocs = []
    for i, structure in enumerate(new_structures):
        fid = i + struct_id
        supposed_name = project_name + f"_iter_{iter_id}_enum_{fid}" + "_static"
        try:
            data = store.query_one({"name": supposed_name}, load=True)
            doc = parse_obj_as(TaskDocument, data)
        except Exception:
            doc = None
        taskdocs.append(doc)

    print("Parsing task documents!")
    parse_output = parse_calculations(taskdocs, enum_output, document)
    print("Fitting calculations!")
    fit_output = fit_calculations(parse_output, document)
    print("Updating output document!")
    new_document = update_document(enum_output, parse_output, fit_output, document)
    new_data = jsanitize(new_document, strict=True, enum_values=True)
    with open("document.json", "w") as fout:
        json.dump(new_data, fout)
    print("Updated document saved! Check with generate.py!")


if __name__ == "__main__":
    __main__()
