const int buzzerPin = 11;
const int btnsPin[8] = {2, 3, 4, 5, 7, 8, 9, 10};

const uint32_t freqMap[16] = {261, 294, 330, 349, 392, 440, 494, 523, 587, 659, 698, 783, 880, 988, 1046};

void setup() {
    Serial.begin(9600);
    for (int i = 0; i < 8; i++) pinMode(btnsPin[i], INPUT);
}

uint16_t calcFreq(uint8_t btns, uint16_t pressure) {
  uint16_t res = 0;
  uint8_t keyCnt = 0;
  for (int i = 0; i < 8; i++) if (btns & (1u << i)) 
    res += freqMap[i];
  res += map(constrain(pressure, 0, 6000), 0, 6000, 0, 100);
  return res;
}

void loop() {
    uint8_t btns = 0;
    for (int i = 0; i < 8; i++) {
        if (digitalRead(btnsPin[i])) {
            btns |= (1u << i);
        }
    }
    if (btns == 0) noTone(buzzerPin);
    else tone(buzzerPin, calcFreq(btns, 0));
    delay(10);
}