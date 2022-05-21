#!/bin/bash


python3 streaming/stream.py & { sleep 1; ngrok http 5014; } & { sleep 5; python3 urlPublish.py; }


# python3 streaming/stream.py & { gnome-terminal; ngrok http 5014; } & { gnome-terminal; python3 real_time_bot_recognition.py; }

echo Done!