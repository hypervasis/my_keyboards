#include QMK_KEYBOARD_H

#define TO_NAV MO(L_NAV)

enum my_layers {
    L_QWERTY,
    /*
      COLEMAK,
      DVORAK,
    */
    L_NAV,
    L_LIGHT,
    L_MACROS,
    L_MOUSE
};
