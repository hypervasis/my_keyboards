#include "../../config.h"

// Disable all Animations to save firmware memory!
// List of animation: https://docs.qmk.fm/#/feature_rgblight?id=effect-and-animation-toggles
#undef RGBLIGHT_ANIMATIONS
#define RGBLIGHT_EFFECT_STATIC_GRADIENT
// #define RGBLIGHT_EFFECT_RAINBOW_SWIRL

// Set leader key timeout to 0.8 secs
#define LEADER_TIMEOUT 800

// reset the leader key timeout after each key press
#define LEADER_PER_KEY_TIMING
