
#include QMK_KEYBOARD_H
#include "customLogic.h"
#include "keymap.h"

layer_state_t layer_state_set_user(layer_state_t state) {
    switch (get_highest_layer(state)) {
        case L_QWERTY:
            // rgblight_mode(9);
            backlight_disable();
            break;
        case L_NAV:
            // rgblight_mode(29);
            backlight_enable();
            break;
        case L_LIGHT:
            rgblight_mode(26);
            break;
        case L_MACROS:
            rgblight_mode(1);
            break;
        case L_MOUSE:
            rgblight_mode(25);
            break;
    }
    return state;
}
