#import pyshark
#import scapy
from scapy.all import *

#from nrf.nrf_sniffer_ble import sniffer_capture
"""
sniffer = sniffer_capture(interface="COM3-None",
                          baudrate=None,
                          fifo= r"\\.\pipe\scapymb1KKX",
                          control_in=None,
                          control_out=None
                          )

"""
#count = 50
#conf.layers.filter([Raw])

#capture = sniff(iface="nRF Sniffer for Bluetooth LE COM3", prn=lambda x:x.summary(), count=count)


load_extcap()
show_interfaces()
get_if_list()

capture = sniff(iface="nRF Sniffer for Bluetooth LE COM8", timeout=5, prn=lambda x: x.summary())

print("\n")
print("capture complete")
print("\n")

print("Number of packets captured: ", len(capture))
print("\n")

for idx, pkt in enumerate(capture):
    val = hexstr(pkt, onlyhex=1)
    val = val.replace(" ", "")

    print(val)
