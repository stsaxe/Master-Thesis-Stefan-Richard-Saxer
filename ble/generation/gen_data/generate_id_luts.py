import yaml
import json
import gzip
import urllib.request

from base64 import b85encode

def fetch_company_id():
    URL = "https://bitbucket.org/bluetooth-SIG/public/raw/main/assigned_numbers/company_identifiers/company_identifiers.yaml"

    with urllib.request.urlopen(URL) as stream:
        DATA = yaml.safe_load(stream.read())

    COMPILED = {}

    for company in DATA["company_identifiers"]:
        COMPILED[company["value"]] = company["name"]

    COMPILED = gzip.compress(json.dumps(COMPILED).encode())
    COMPILED = b85encode(COMPILED).decode()
    COMPILED = "\n".join(COMPILED[i : i + 79] for i in range(0, len(COMPILED), 79)) + "\n"


    with open("BluetoothIDs.py", "r") as inp:
        data = inp.read()

    with open("BluetoothIDs.py", "w") as out:
        ini, sep, _ = data.partition("COMPANY_IDs = _d(\"\"\"")
        COMPILED = ini + sep + "\n" + COMPILED + "\"\"\")\n"
        print("Written: %s" % out.write(COMPILED))


def fetch_uuid():
    URL = "https://bitbucket.org/bluetooth-SIG/public/raw/main/assigned_numbers/uuids/member_uuids.yaml"

    with urllib.request.urlopen(URL) as stream:
        DATA = yaml.safe_load(stream.read())

    COMPILED = {}

    for company in DATA["uuids"]:
        COMPILED[company["uuid"]] = company["name"]

    COMPILED = gzip.compress(json.dumps(COMPILED).encode())
    COMPILED = b85encode(COMPILED).decode()
    COMPILED = "\n".join(COMPILED[i: i + 79] for i in range(0, len(COMPILED), 79)) + "\n"

    with open("BluetoothIDs.py", "r") as inp:
        data = inp.read()

    with open("BluetoothIDs.py", "w") as out:
        ini, sep, _ = data.partition("UUIDs = _d(\"\"\"")
        COMPILED = ini + sep + "\n" + COMPILED + "\"\"\")\n"
        print("Written: %s" % out.write(COMPILED))


if __name__ == "__main__":
    #fetch_company_id()
    #fetch_uuid()
    pass
