#!/bin/python3

import hid

# Define a list of device params, for which we must connect
vid = 0xf1f1
pid = 0x0315
usage_page = 0xff60
usage_id = 0x61


# for device in hid.enumerate():
#     if device['vendor_id'] == vid and device['product_id'] == pid:
#         print(device)

for device in hid.enumerate():
    if device['vendor_id'] == vid and device['product_id'] == pid and device['usage_page'] == usage_page and device['usage'] == usage_id:
        print(f"Opening keyboard: {device['path']}")
        keyboard = hid.Device(path=device['path'])
        break


# to_send = pad_message(b'1')
# print(f"To send : {to_send}")
byte_array = bytearray();
byte_array.append(0)
byte_array.append(0)
keyboard.write(bytes(byte_array))
keyboard.close()
