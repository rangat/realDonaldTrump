from slackclient import SlackClient
BOT_NAME = 'readldonalddrumpf'
slack = SlackClient('api_key')

botid = 'U2A7XR8I1'
chan ="C2A7YGWRM"
print slack.api_call("auth.test")
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
                 print (message)
                 print (message.get("user"))
                 if(message.get("text") is None):
                     print("NONE")
                 if(message.get("text") is not  None and message.get("user") !=  "U2A7XR81X"):
                    slack.api_call("chat.postMessage", as_user="true", channel=chan, text=message.get("text"))
                
