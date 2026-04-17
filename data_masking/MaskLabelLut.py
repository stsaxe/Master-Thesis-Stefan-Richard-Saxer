from data_processing.LabelLut import (LABEL_APPLE_FIND_MY,
                                      LABEL_APPLE_FIND_MY_GEN_2,
                                      LABEL_GOOGLE_FIND_MY,
                                      Label_SAMSUNG_SMART_THINGS,
                                      LABEL_DULT,
                                      LABEL_TILE_NETWORK,
                                      LABEl_APPLE_DEVICE,
                                      LABEl_APPLE_FIND_MY_DEVICE,
                                      LABEL_OTHER_DEVICE,
                                      STATE_NEARBY,
                                      STATE_LOST,
                                      STATE_SEARCHING,
                                      STATE_UNPAIRED,
                                      STATE_OFFLINE,
                                      STATE_ONLINE,
                                      SEPARATOR,
                                      LABEL_LUT_PROD
                                      )

MASK_LABEL_LUT = {
    LABEL_OTHER_DEVICE: "mask_config_default.yaml",

    LABEl_APPLE_DEVICE: "mask_config_apple.yaml",

    LABEL_APPLE_FIND_MY + SEPARATOR + STATE_LOST: "mask_config_apple.yaml",
    LABEL_APPLE_FIND_MY + SEPARATOR + STATE_NEARBY: "mask_config_apple.yaml",
    LABEL_APPLE_FIND_MY + SEPARATOR + STATE_UNPAIRED: "mask_config_apple.yaml",

    LABEl_APPLE_FIND_MY_DEVICE + SEPARATOR + STATE_OFFLINE: "mask_config_apple.yaml",
    LABEl_APPLE_FIND_MY_DEVICE + SEPARATOR + STATE_ONLINE: "mask_config_apple.yaml",

    LABEL_APPLE_FIND_MY_GEN_2 + SEPARATOR + STATE_LOST: "mask_config_apple_find_my_gen_2.yaml",
    LABEL_APPLE_FIND_MY_GEN_2 + SEPARATOR + STATE_NEARBY: "mask_config_apple_find_my_gen_2.yaml",

    LABEL_GOOGLE_FIND_MY + SEPARATOR + STATE_LOST: "mask_config_google_find_my.yaml",
    LABEL_GOOGLE_FIND_MY + SEPARATOR + STATE_NEARBY: "mask_config_google_find_my.yaml",
    LABEL_GOOGLE_FIND_MY + SEPARATOR + STATE_UNPAIRED: "mask_config_default.yaml",

    Label_SAMSUNG_SMART_THINGS + SEPARATOR + STATE_LOST: "mask_config_samsung_smart_things.yaml",
    Label_SAMSUNG_SMART_THINGS + SEPARATOR + STATE_NEARBY: "mask_config_samsung_smart_things.yaml",
    Label_SAMSUNG_SMART_THINGS + SEPARATOR + STATE_UNPAIRED: "mask_config_samsung_smart_things.yaml",
    Label_SAMSUNG_SMART_THINGS + SEPARATOR + STATE_SEARCHING: "mask_config_samsung_smart_things.yaml",

    LABEL_TILE_NETWORK + SEPARATOR + STATE_LOST: "mask_config_tile.yaml",
    LABEL_TILE_NETWORK + SEPARATOR + STATE_NEARBY: "mask_config_tile.yaml",
    LABEL_TILE_NETWORK + SEPARATOR + STATE_UNPAIRED: "mask_config_tile.yaml",
    LABEL_TILE_NETWORK + SEPARATOR + STATE_SEARCHING: "mask_config_tile.yaml",

    LABEL_DULT + SEPARATOR + STATE_LOST: "mask_config_dult.yaml",
    LABEL_DULT + SEPARATOR + STATE_NEARBY: "mask_config_dult.yaml",
}



for key, value in LABEL_LUT_PROD.items():
    assert value in MASK_LABEL_LUT, f"key {key} not in MASK_LABEL_LUT"


