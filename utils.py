import numpy as np
import libinsitu
import pvlib
import sg2
import requests
from xml.etree import ElementTree as ET


from libinsitu import netcdf_to_dataframe, visual_qc
from datetime import datetime

BASE_URL = "http://tds.webservice-energy.org/"
INSITU_CQTQLOGUE_URL = BASE_URL + "thredds/in-situ.xml"

namespaces = {
    "thredds": "http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0"
}

# Authorized subcatalogs (Replace with your actual list)
authorized_subcatalogs = [
    "IEA-PVPS",
    "SURFRAD",
    "BSRN",
    "ESMAP",
    "NREL-MIDC",
    "SOLRAD",
]


import requests
from requests.auth import HTTPBasicAuth

def display_subcatalog_datasets(username, password):
    # Create a session with authentication
    session = requests.Session()
    session.auth = HTTPBasicAuth(username, password)

    # Fetch and parse the in-situ catalog
    response = session.get(INSITU_CQTQLOGUE_URL)
    response.raise_for_status()
    root = ET.fromstring(response.content)

    # Find all subcatalogs in the in-situ catalog
    subcatalog_links = [
        cat.get("{http://www.w3.org/1999/xlink}href")
        for cat in root.findall(".//thredds:catalogRef", namespaces)
    ]
    subcatalog_names = [
        cat.get("{http://www.w3.org/1999/xlink}title")
        for cat in root.findall(".//thredds:catalogRef", namespaces)
    ]

    dataset_info = {}

    # Iterate through each subcatalog to retrieve .nc file URLs
    for subcat_name, subcat_link in zip(subcatalog_names, subcatalog_links):
        if subcat_name in authorized_subcatalogs:
            # Fetch and parse the authorized subcatalog
            if subcat_link.startswith(BASE_URL):
                subcat_url = subcat_link
            else:
                subcat_url = BASE_URL[:-1] + subcat_link
            response = session.get(subcat_url)
            response.raise_for_status()
            subcat_root = ET.fromstring(response.content)

            # Extract .nc dataset URLs
            datasets = [
                ds.get("urlPath")
                for ds in subcat_root.findall(".//thredds:dataset", namespaces)
                if ds.get("urlPath") and ds.get("urlPath").endswith(".nc")
            ]
            if datasets:
                dataset_info[subcat_name] = datasets
    return dataset_info



from netCDF4 import Dataset
from urllib.parse import urlsplit, quote_plus
import xarray as xr
import pandas as pd 

def openNetCDF(filename, mode='r', user=None, password=None):
    """ Open either a filename or OpenDAP URL with user/password"""
    if '://' in filename:
        if user:
            filename = with_auth(filename, user, password)
        filename = "[FillMismatch]" + filename
    return Dataset(filename, mode=mode)

def with_auth(url, user, password):
    parts = urlsplit(url)
    return "%s://%s:%s@%s/%s" % (parts.scheme, quote_plus(user), quote_plus(password), parts.netloc, parts.path)