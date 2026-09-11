# ESP32 firmware status

The exact seven-channel BLE firmware used during the glove experiments has not yet been located in the archived working files and is therefore not included.

The released host scripts document the interface they received:

- Nordic UART Service TX characteristic: 6E400003-B5A3-F393-E0A9-E50E24DCCA9E
- notification payload: device time followed by seven raw ADC values
- channel order: GPIO36, GPIO39, GPIO34, GPIO35, GPIO32, GPIO33, GPIO25
- ADC configuration recorded in the host-side methods: 12-bit resolution, 11-dB attenuation, eight readings averaged per channel, approximately 20 Hz output

An unrelated six-channel, 5 V serial Arduino sketch using a 7.5 kΩ divider was found but deliberately excluded because it is not the glove BLE firmware. A verified firmware file should be added only after it is recovered from the study ESP32 project or the original computer.
