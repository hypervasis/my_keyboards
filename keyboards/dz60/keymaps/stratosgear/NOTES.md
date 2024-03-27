# Notes

From the root of the project run:

`qmk flash` 

It will complain that:

`ERROR: Bootloader not found. Trying again in 5s.`

Which means:
* Unplug keyboard
* Press SPC+B then plug in (release the keys after a couple of secs after plug in)
* The flash should take place

You can also run:

`qmk compile`

to only compile the firmware.

In order to run these "short" commands it assumes that you have already
configured the prefered keyboards and keymaps to work with.

This is done with:

```
qmk config user.keyboard=dz60
qmk config user.keymap=stratosgear
```

which are saved to `/home/stratos/.config/qmk/qmk.ini`
