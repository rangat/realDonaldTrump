from slackclient import SlackClient
import question_similarity as q
BOT_NAME = 'FDR'
slack = SlackClient('api-key')

botid = 'U2A7XR8I1'
chan ="C2A7YGWRM"
def channel_info(channel_id):
        channel_info = slack.api_call("channels.info", channel=channel_id)
        if channel_info:
             return channel_info['channel']
             return None
def list_channels():
     channels_call = slack.api_call("channels.list")
     if channels_call.get('ok'):
         return channels_call['channels']
channels = list_channels()

#sends message to a channel
if(slack.rtm_connect()):
        while True:
             mes = slack.rtm_read()
             for message in mes:
                 print(message)
                 print (message.get("user"))
                 if(message.get("text") is None):
                     print("NONE")

                 if(message.get("text") is not  None and message.get("user") !=  "U2AB9LP29"):
                     mes = q.get_answer("SANDERS",'Sanders',message.get("text"))
                     slack.api_call("chat.postMessage", as_user="true", channel=message.get("channel"), text=mes)
                
