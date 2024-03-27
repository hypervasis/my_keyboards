
#include QMK_KEYBOARD_H
#include "raw_hid.h"

// Defines names for use in layer keycodes and the keymap

#define layers_count 2
enum layer_names {
    _BASE,
    _OBS,
};

enum custom_keycodes {
    CK_OBS_CAMERA = SAFE_RANGE,
};

// Keycodes that are emitted whenever the right encoder is rotated to indicate the layer selected
uint16_t layer_emit_keys[] = {KC_F13, KC_F14};

// clang-format off
const uint16_t PROGMEM keymaps[][MATRIX_ROWS][MATRIX_COLS] = {
    // Base
    [_BASE] = LAYOUT_top(
               KC_HOME, KC_F13, CK_OBS_CAMERA,
        KC_1,   KC_2,    KC_3,   KC_4,          KC_5,
        KC_6,  KC_7,    KC_8,   KC_9,          KC_0,
        KC_NO, KC_NO,   KC_NO,  KC_NO,         QK_BOOT
    ),

    // OBS Studio (Assigned F14 for most shortcut actions)
    [_OBS] = LAYOUT_top(
                      KC_HOME,         KC_F14,         KC_TRNS,
        C(KC_F14),    C(S(KC_F14)),    C(G(KC_F14)),    C(S(G(KC_F14))),    S(KC_F14),
        A(KC_F14),    A(S(KC_F14)),    A(G(KC_F14)),    A(S(G(KC_F14))),    G(KC_F14),
        C(A(KC_F14)), C(A(S(KC_F14))), C(A(G(KC_F14))), C(A(S(G(KC_F14)))), KC_TRNS
    ),

    // Empty pattern that fully utilizes ALL possible MOD keys with an F13-F24 key!!!
    // [_LAYER_NAME] = LAYOUT_top(
    //                   KC_TRNS,         KC_F1X,          KC_TRNS,
    //     C(KC_F1X),    C(S(KC_F1X)),    C(G(KC_F15)),    C(S(G(KC_F1X))),    S(KC_F1X),
    //     A(KC_F1X),    A(S(KC_F1X)),    A(G(KC_F18)),    A(S(G(KC_F1X))),    G(KC_F1X),
    //     C(A(KC_F1X)), C(A(S(KC_F1X))), C(A(G(KC_F21))), C(A(S(G(KC_F1X)))), S(G(KC_F1X))
    // ),

};
// clang-format on
uint8_t selected_layer = 0;
uint8_t selected_obs_scene = 0;
uint16_t obs_scene_keycodes[] = {C(G(KC_F14)), C(A(G(KC_F14))), C(A(KC_F14)), C(KC_F14)};
bool encoder_update_user(uint8_t index, bool clockwise) {
    uint8_t layer = get_highest_layer(layer_state | default_layer_state);
    xprintf("Encoder: %d\n", index);

    switch (index) {
        case 0:
            // Left encoder
            if (layer == _OBS) {
                if (clockwise) {
                    rgblight_decrease_hue();
                } else {
                    rgblight_increase_hue();
                }
            } else {
                tap_code(clockwise ? KC_PGDN : KC_PGUP);
            }
            break;

        case 1:
            // Center encoder
            if (clockwise) {
                if (selected_layer == layers_count-1) {
                    selected_layer = 0;
                } else {
                    selected_layer++;
                }
            } else {
                if (selected_layer == 0) {
                    selected_layer = layers_count-1;
                } else {
                    selected_layer--;
                }
            }
            xprintf("Selected Layer: %d\n", selected_layer);
            layer_clear();
            layer_on(selected_layer);
            tap_code(layer_emit_keys[selected_layer]);
            break;

        case 2:
            // Right encoder
            // if (layer == _OBS) {
            if (clockwise) {
                if (selected_obs_scene == 3) {
                    selected_obs_scene = 0;
                } else {
                    selected_obs_scene++;
                }
            } else {
                if (selected_obs_scene == 0) {
                    selected_obs_scene = 3;
                } else {
                    selected_obs_scene--;
                }
            }
            tap_code16(obs_scene_keycodes[selected_obs_scene]);
            // } else {
            //     tap_code_delay(clockwise ? KC_MNXT : KC_MPRV, 10);
            // }
            break;
    }
    return false;
}

bool obs_full_face = true;

bool process_record_user(uint16_t keycode, keyrecord_t *record) {
    switch (keycode) {
    case CK_OBS_CAMERA:
        if (record->event.pressed) {
            // when keycode CK_OBS_CAMERA is pressed
            obs_full_face = !obs_full_face;
            if (obs_full_face) {
                tap_code16(C(S(KC_F14)));
            } else {
                tap_code16(A(S(KC_F14)));
            }
        } else {
            // when keycode CK_OBS_CAMERA is released
        }
        break;
    }
    return true;
};

#ifdef CONSOLE_ENABLE
void keyboard_post_init_user(void) {
  // Customise these values to desired behaviour
  debug_enable=true;
  //debug_matrix=true;
  debug_keyboard=true;
  //debug_mouse=true;
}
#endif


#ifdef RAW_ENABLE
void raw_hid_receive(uint8_t* data, uint8_t length) {

    xprintf("Externally setting layer: %d\n", data[0]);
    layer_clear();
    layer_on(data[0]);
}
#endif


//                  F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24
// A(G(KC))                                                      x
// A(KC)                                                         x
// A(S(G(KC)))                                                   x
// A(S(KC))                                                      x
// C(A(G(KC)))           x                                       x
// C(A(KC))              x                                       x
// C(A(S(G(KC))))                                                x
// C(A(S(KC)))                                                   x
// C(G(KC))              x                                       x
// C(KC)                 x                                       x
// C(S(G(KC)))                                                   x
// C(S(KC))                                                      x
// G(KC)                                                         x
// S(KC)                                                         x

