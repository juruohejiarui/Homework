#define SensorPin A0
const int buzzerPin = 11;
const uint32_t forceConstant = 510;
const uint8_t btnsPin[8] = {2, 3, 4, 5, 7, 8, 9, 10};

const uint32_t freqMap[16] = {261, 294, 330, 349, 392, 440, 494, 523, 587, 659, 698, 783, 880, 988, 1046};

inline uint32_t calcPressure(uint32_t fsr) {
  return (uint32_t)(1023 - fsr) * forceConstant / fsr;
}

uint32_t calcFreq(uint8_t btns, uint32_t pressure) {
  uint32_t res = 0;
  for (int i = 0; i < 8; i++) if (btns & (1u << i)) 
    res += freqMap[i];
  if (pressure > 10)
    res = res * (950 + map(constrain(pressure, 0, 70000), 0, 70000, 0, 100)) / 1000;
  return res;
}

void setup() {
  Serial.begin(9600);
  for (int i = 0; i < 8; i++) pinMode(btnsPin[i], INPUT);
  pinMode(buzzerPin, OUTPUT);
}

void loop() {
  uint32_t fsr = calcPressure(analogRead(SensorPin));
  uint8_t btnState = 0;
  for (int i = 0; i < 8; i++) btnState |= (digitalRead(btnsPin[i]) << i);
  if (!btnState) { noTone(buzzerPin); delay(10); }
  else {
    Serial.println(fsr);
    tone(buzzerPin, calcFreq(btnState, fsr));
    delay(10);
  }
}