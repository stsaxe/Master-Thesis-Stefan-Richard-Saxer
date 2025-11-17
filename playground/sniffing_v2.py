from scapy.all import *
from scapy.layers.bluetooth4LE import BTLE
from scapy.contrib import *

load_contrib("nrf_sniffer")
load_extcap()

file = PcapReader(r"C:\Users\stsax\OneDrive\Studium\9. Semester\Masterarbeit\Repository\data\pcap\AirTag\AirTag_(nearby)_3h.pcapng")

for idx, pkt in enumerate(file):
    val = hexstr(pkt, onlyhex=1)
    val = val.replace(" ", "")
    val_slice = val[34:]

    #print(type(val_slice))
    p = BTLE(hex_bytes(val_slice))
    print(p.show())

    print(pkt.show())

    new_p = scapy.contrib.nrf_sniffer.NRF2_Packet_Event(hex_bytes(val))
    print(new_p.show())

    load = new_p['Raw'].load
    bitstring = load.hex()
    print(bitstring)




    #print(p.summary())

    #bitstring = load.hex()
    #print(bitstring)

    break

"""

p = BTLE(hex_bytes('d6be898e4024320cfb574d5a02011a1aff4c000c0e009c6b8f40440f1583ec895148b410050318c0b525b8f7d4'))
#print(p.show())

#p['BTLE ADV_IND']['Raw'].load = b"\x00"
#p['BTLE ADV_IND']['EIR Manufacturer Specific Data'].company_id = 77

print(p.show())
p['BTLE ADV_IND']['EIR Manufacturer Specific Data'].company_id = 77
load = p['BTLE ADV_IND']['Raw'].load
p['BTLE ADV_IND']['Raw'].load = b"\x00"
del p['BT4LE'].crc
del p['BTLE advertising header'].Length
#print(l1.show())
#l1['EIR Manufacturer Specific Data'].company_id = 77

#l1['Raw'].load = b"\x00"
#print(l1.show())
print(p.show())

p = BTLE(bytes(p))

print(p.show())

bitstring = load.hex()
print(bitstring)

"""
