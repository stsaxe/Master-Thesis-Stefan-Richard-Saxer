LABEL_APPLE_FIND_MY = "Apple Find My Tracker"
LABEL_APPLE_FIND_MY_GEN_2 = "Apple Find My Tracker Gen 2"

LABEL_GOOGLE_FIND_MY = "Google Find My Tracker"
Label_SAMSUNG_SMART_THINGS = "Samsung SmartThings Tracker"
LABEL_TILE_NETWORK = "Tile Tracker"

LABEL_DULT = "DULT"

LABEl_APPLE_DEVICE = "Apple Device"
LABEl_APPLE_FIND_MY_DEVICE = "Apple Find My Device"

LABEL_OTHER_DEVICE = "Other Device"

STATE_NEARBY = "(nearby)"
STATE_LOST = "(lost)"
STATE_SEARCHING = "(searching)"
STATE_UNPAIRED = "(unpaired)"
STATE_OFFLINE = "(offline)"
STATE_ONLINE = "(online)"

SEPARATOR = " "

LABEL_LUT_PROD = {
    '4Smarts SkyTag (lost)': LABEL_APPLE_FIND_MY + SEPARATOR + STATE_LOST,
    '4Smarts SkyTag (nearby)': LABEL_APPLE_FIND_MY + SEPARATOR + STATE_NEARBY,
    '4Smarts SkyTag (unpaired)': LABEL_APPLE_FIND_MY + SEPARATOR + STATE_UNPAIRED,

    '4Smarts SkyTag Card (lost)': LABEL_APPLE_FIND_MY + SEPARATOR + STATE_LOST,
    '4Smarts SkyTag Card (nearby)': LABEL_APPLE_FIND_MY + SEPARATOR + STATE_NEARBY,
    '4Smarts SkyTag Card (unpaired)': LABEL_APPLE_FIND_MY + SEPARATOR + STATE_UNPAIRED,

    'Apple AirPod CT 07 (lost)': LABEl_APPLE_DEVICE,
    'Apple AirPod CT 07 (nearby)': LABEl_APPLE_DEVICE,
    'Apple AirPod FindMy offline (lost)': LABEL_APPLE_FIND_MY + SEPARATOR + STATE_LOST,
    'Apple AirPod FindMy online (nearby)': LABEL_APPLE_FIND_MY + SEPARATOR + STATE_NEARBY,

    'Apple AirTag (lost)': LABEL_APPLE_FIND_MY + SEPARATOR + STATE_LOST,
    'Apple AirTag (nearby)': LABEL_APPLE_FIND_MY + SEPARATOR + STATE_NEARBY,
    'Apple AirTag (unpaired)': LABEL_APPLE_FIND_MY + SEPARATOR + STATE_UNPAIRED,

    'Apple AirTag 2 (lost)': LABEL_APPLE_FIND_MY_GEN_2 + SEPARATOR + STATE_LOST,
    'Apple AirTag 2 (nearby)': LABEL_APPLE_FIND_MY_GEN_2 + SEPARATOR + STATE_NEARBY,
    'Apple AirTag 2 (unpaired)': LABEL_APPLE_FIND_MY + SEPARATOR + STATE_UNPAIRED,

    'Apple MacBook CT 09 (online)': LABEl_APPLE_DEVICE,
    'Apple MacBook CT 10 (offline)': LABEl_APPLE_DEVICE,
    'Apple MacBook CT 10 (online)': LABEl_APPLE_DEVICE,
    'Apple MacBook CT 16 (offline)': LABEl_APPLE_DEVICE,
    'Apple MacBook FindMy offline (offline)': LABEl_APPLE_FIND_MY_DEVICE + SEPARATOR + STATE_OFFLINE,
    'Apple MacBook FindMy online (online)': LABEl_APPLE_FIND_MY_DEVICE + SEPARATOR + STATE_ONLINE,

    'Apple iPad CT 10 (offline)': LABEl_APPLE_DEVICE,
    'Apple iPad CT 10 (online)': LABEl_APPLE_DEVICE,
    'Apple iPad FindMy offline (offline)': LABEl_APPLE_FIND_MY_DEVICE + SEPARATOR + STATE_OFFLINE,
    'Apple iPad FindMy online (online)': LABEl_APPLE_FIND_MY_DEVICE + SEPARATOR + STATE_ONLINE,

    'Apple iPhone CT 10 (offline)': LABEl_APPLE_DEVICE,
    'Apple iPhone CT 10 (online)': LABEl_APPLE_DEVICE,
    'Apple iPhone CT 16 (offline)': LABEl_APPLE_DEVICE,
    'Apple iPhone CT 16 (online)': LABEl_APPLE_DEVICE,
    'Apple iPhone FindMy offline (offline)': LABEl_APPLE_FIND_MY_DEVICE + SEPARATOR + STATE_OFFLINE,
    'Apple iPhone FindMy online (online)': LABEl_APPLE_FIND_MY_DEVICE + SEPARATOR + STATE_ONLINE,

    'Chipolo CARD [Apple] (lost)': LABEL_APPLE_FIND_MY + SEPARATOR + STATE_LOST,
    'Chipolo CARD [Apple] (nearby)': LABEL_APPLE_FIND_MY + SEPARATOR + STATE_NEARBY,
    'Chipolo CARD [Apple] (unpaired)': LABEL_APPLE_FIND_MY + SEPARATOR + STATE_UNPAIRED,

    'Chipolo CARD [Google] (lost)': LABEL_GOOGLE_FIND_MY + SEPARATOR + STATE_LOST,
    'Chipolo CARD [Google] (nearby)': LABEL_GOOGLE_FIND_MY + SEPARATOR + STATE_NEARBY,
    'Chipolo CARD [Google] (unpaired)': LABEL_GOOGLE_FIND_MY + SEPARATOR + STATE_UNPAIRED,

    'Chipolo ONE (lost)': LABEL_APPLE_FIND_MY + SEPARATOR + STATE_LOST,
    'Chipolo ONE (nearby)': LABEL_APPLE_FIND_MY + SEPARATOR + STATE_NEARBY,
    'Chipolo ONE (unpaired)': LABEL_APPLE_FIND_MY + SEPARATOR + STATE_UNPAIRED,

    'Hama MGF (lost)': LABEL_GOOGLE_FIND_MY + SEPARATOR + STATE_LOST,
    'Hama MGF (nearby)': LABEL_GOOGLE_FIND_MY + SEPARATOR + STATE_NEARBY,
    'Hama MGF (unpaired)': LABEL_GOOGLE_FIND_MY + SEPARATOR + STATE_UNPAIRED,

    'KeySmart SmartCard (lost)': LABEL_APPLE_FIND_MY + SEPARATOR + STATE_LOST,
    'KeySmart SmartCard (nearby)': LABEL_APPLE_FIND_MY + SEPARATOR + STATE_NEARBY,
    'KeySmart SmartCard (unpaired)': LABEL_APPLE_FIND_MY + SEPARATOR + STATE_UNPAIRED,

    'Lifemate LifeTag (lost)': LABEL_GOOGLE_FIND_MY + SEPARATOR + STATE_LOST,
    'Lifemate LifeTag (nearby)': LABEL_GOOGLE_FIND_MY + SEPARATOR + STATE_NEARBY,
    'Lifemate LifeTag (unpaired)': LABEL_GOOGLE_FIND_MY + SEPARATOR + STATE_UNPAIRED,

    'Motorola MotoTag (lost)': LABEL_GOOGLE_FIND_MY + SEPARATOR + STATE_LOST,
    'Motorola MotoTag (nearby)': LABEL_GOOGLE_FIND_MY + SEPARATOR + STATE_NEARBY,
    'Motorola MotoTag (unpaired)': LABEL_GOOGLE_FIND_MY + SEPARATOR + STATE_UNPAIRED,

    'Samsung SmartTag (lost)': Label_SAMSUNG_SMART_THINGS + SEPARATOR + STATE_LOST,
    'Samsung SmartTag (nearby)': Label_SAMSUNG_SMART_THINGS + SEPARATOR + STATE_NEARBY,
    'Samsung SmartTag (searching)': Label_SAMSUNG_SMART_THINGS + SEPARATOR + STATE_SEARCHING,
    'Samsung SmartTag (unpaired)': Label_SAMSUNG_SMART_THINGS + SEPARATOR + STATE_UNPAIRED,

    'Tile Mate (lost)': LABEL_TILE_NETWORK + SEPARATOR + STATE_LOST,
    'Tile Mate (nearby)': LABEL_TILE_NETWORK + SEPARATOR + STATE_NEARBY,
    'Tile Mate (searching)': LABEL_TILE_NETWORK + SEPARATOR + STATE_SEARCHING,
    'Tile Mate (unpaired)': LABEL_TILE_NETWORK + SEPARATOR + STATE_UNPAIRED,

    'Tile Slim (lost)': LABEL_TILE_NETWORK + SEPARATOR + STATE_LOST,
    'Tile Slim (nearby)': LABEL_TILE_NETWORK + SEPARATOR + STATE_NEARBY,
    'Tile Slim (unpaired)': LABEL_TILE_NETWORK + SEPARATOR + STATE_UNPAIRED,

    'other Device': LABEL_OTHER_DEVICE

}

