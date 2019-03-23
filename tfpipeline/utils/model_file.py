from oauth2client.client import GoogleCredentials
from googleapiclient import discovery
import json

client = discovery.build('storage', 'v1') # add http=whatever param if auth
request = client.objects().list(
    bucket="imagerie3",
    prefix="trained_models/",
    delimiter="/")

response = request.execute()
ptxt = 'model_config_list {\n'
for s in response['prefixes']:
    s = s[len('trained_models/'):]

    ptxt +="\tconfig {\n"
    ptxt +="\t\tname: '{}'\n".format(s[:len(s)-1])
    ptxt +="\t\tbase_path: 's3://{}/trained_models/{}'\n".format("imagerie3",s)
    ptxt +="\t\tmodel_platform: \'tensorflow\'\n"
    ptxt +="\t\tmodel_version_policy: {all: {}}\n"
    ptxt +='\t}\n'


ptxt +="}"
print(ptxt)