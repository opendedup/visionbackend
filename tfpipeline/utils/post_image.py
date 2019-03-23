import requests
import base64
import sys
import json


def main():
    # print command line arguments
    for arg in sys.argv[1:]:
        with open(arg, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            imgjs = {'b64':encoded_string}
            d=json.dumps({'instances':[imgjs]})
            print(d[:30])
            r = requests.post("http://localhost:8501/v1/models/model2:predict", data=d)
            print(r.status_code, r.reason)
            print(r.text)

if __name__ == "__main__":
    main()
