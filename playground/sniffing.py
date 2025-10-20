import pyshark
import scapy
from scapy.all import *


from nrf.nrf_sniffer_ble import sniffer_capture
"""
sniffer = sniffer_capture(interface="COM3-None",
                          baudrate=None,
                          fifo= r"\\.\pipe\scapymb1KKX",
                          control_in=None,
                          control_out=None
                          )

"""
load_extcap()

show_interfaces()

get_if_list()

count = 50
conf.layers.filter([Raw])

#capture = sniff(iface="nRF Sniffer for Bluetooth LE COM3", prn=lambda x:x.summary(), count=count)
capture = sniff(iface="nRF Sniffer for Bluetooth LE COM3", timeout = 5)




print("capture complete")

print(len(capture))
start_time = 0
end_time = 0
for idx, pkt in enumerate(capture):
    val = hexstr(pkt, onlyhex=1)
    val = val.replace(" ", "")
    time = pkt.time


    if idx == time:
        start_time = time




