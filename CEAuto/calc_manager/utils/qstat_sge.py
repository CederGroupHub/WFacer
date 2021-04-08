"""Parse qstat output of SGE."""

import subprocess as sp
import xmltodict


def qstat(qstat_path='qstat', xml_option='-xml'):
    """
    Parameters
    ----------
    qstat_path : string
        The path to the qstat executable.

    Returns
    -------
    queue_info : list
        A list of jobs in 'queue_info'. Jobs are dictionaries with both string 
        keys and string names.
    job_info : list
        A list of jobs in 'job_info'.
    """
    xml = qstat2xml(qstat_path=qstat_path)
    return xml2queue_and_job_info(xml)


def qstat2xml(qstat_path='qstat', xml_option='-xml'):
    """
    Parameters
    ----------
    qstat_path : string
        The path to the qstat executable.

    Returns
    -------
    qstatxml : string
        The xml stdout string of the 'qstat -xml' call.
    """
    try:
        qstatxml = sp.check_output([qstat_path, xml_option], stderr=sp.STDOUT)
    except sp.CalledProcessError as e:
        print('qstat returncode:', e.returncode)
        print('qstat std output:', e.output)
        raise
    except FileNotFoundError as e:
        e.message = ('Maybe "' + qstat_path + ' ' +
                     xml_option + '" is not installed.')
        raise
    return qstatxml


def xml2queue_and_job_info(qstatxml):
    """
    Parameters
    ----------
    qstatxml : string
        The xml string of the 'qstat -xml' call.

    Returns
    -------
    queue_info : list
        A list of jobs in 'queue_info'. Jobs are dictionaries with both string 
        keys and string names.
    job_info : list
        A list of jobs in 'job_info'.
    """
    x = xmltodict.parse(qstatxml)
    queue_info = x["job_info"]["queue_info"]
    job_info = x["job_info"]["job_info"]
    queue_dicts = []
    if queue_info is not None:
        if isinstance(queue_info["job_list"], list):
            for dict_ in queue_info["job_list"]:
                queue_dicts.append(dict_)
        else:
            queue_dicts.append(queue_info["job_list"])
    job_dicts = []
    if job_info is not None:
        if isinstance(job_info["job_list"], list):
            for dict_ in job_info["job_list"]:
                job_dicts.append(dict_)
        else:
            job_dicts.append(job_info["job_list"])
    return queue_dicts, job_dicts
