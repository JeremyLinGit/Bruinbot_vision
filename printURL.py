import json
import os 

os.system("curl  http://localhost:4040/api/tunnels > tunnels.json")

with open('tunnels.json') as data_file:    
    datajson = json.load(data_file)

print(datajson['tunnels'][0]['public_url'])


msg = "ngrok URL's: \n"
for i in datajson['tunnels']:
    #print(i)
    msg = msg + i['public_url'] +'\n'

#print (msg.shape)