SRC += features/select_word.c
SRC += features/achordion.c

# Convert from Elite-C to standard RP2040
# CONVERT_TO=promicro_rp2040
CONVERT_TO=rp2040_ce

OLED_ENABLE = yes
# OLED_DRIVER = SSD1306      # Enables the use of OLED displays

ENCODER_ENABLE = yes       # Enables the use of one or more encoders

# BACKLIGHT_ENABLE = yes
RGB_MATRIX_ENABLE = no     # Disable keyboard RGB matrix, as it is enabled by default on rev3
RGBLIGHT_ENABLE = yes      # Enable keyboard RGB underglow

TAP_DANCE_ENABLE = yes

MOUSEKEY_ENABLE = yes

# Allows the creation of encoder keymap arrays.
# ENCODER_MAP_ENABLE = yes

CONSOLE_ENABLE = yes


LEADER_ENABLE = yes

DYNAMIC_MACRO_ENABLE = yes

DYNAMIC_TAPPING_TERM_ENABLE = yes
