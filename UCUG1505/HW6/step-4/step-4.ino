#define SensorPin A0
#define AutoMode
const int buzzerPin = 11;
const int resistence0 = 10000;
const int btnsPin[8] = {2, 3, 4, 5, 7, 8, 9, 10};

// add a filter
#define MidAvgFilterSz  (8u)
template<typename T>
struct MidAvgFilter {
  T buf[MidAvgFilterSz];
  uint8_t idx = 0;
  MidAvgFilter() {
    this->idx = 0;
    memset(this->buf, 0, sizeof(MidAvgFilterSz));
  }
  T getVal(T newData) {
    T tem;
    uint16_t i, j;
    static T tmpArr[MidAvgFilterSz];
    this->buf[this->idx] = newData;
    this->idx = (this->idx + 1) % MidAvgFilterSz;
    memcpy(tmpArr, buf, sizeof(buf));
    for (i = 0; i < MidAvgFilterSz - 1; i++) {
      uint8_t flag = 0;
      for (j = i + 1; j < MidAvgFilterSz; j++)
        if (tmpArr[j] > tmpArr[j + 1]) {
          T tmp = tmpArr[j + 1];
          tmpArr[j + 1] = tmpArr[j];
          tmpArr[j] = tmp;
          flag = 1;
        }
      if (!flag) break;
    }
    if (MidAvgFilterSz & 1)
      return tmpArr[(MidAvgFilterSz + 1) >> 1];
    else return (tmpArr[MidAvgFilterSz >> 1] + tmpArr[(MidAvgFilterSz >> 1) | 1]) >> 1;
  }
};

MidAvgFilter<uint16_t> pressureFilter;

const uint32_t freqMap[16] = {261, 294, 330, 349, 392, 440, 494, 523, 587, 659, 698, 783, 880, 988, 1046};

inline uint16_t calcPressure(uint16_t fsr) {
  return (1023 - fsr) * resistence0 / fsr;
}

uint16_t calcFreq(uint8_t btns, uint16_t pressure) {
  float res = 0;
  uint8_t keyCnt = 0;
  for (int i = 0; i < 8; i++) if (btns & (1u << i)) 
    res += freqMap[i];
  res += map(constrain(pressure, 0, 6000), 0, 6000, 0, 100);
  return res;
}

void setup() {
  Serial.begin(9600);
  for (int i = 0; i < 8; i++) pinMode(btnsPin[i], INPUT);
  pinMode(buzzerPin, OUTPUT);
}

void loop() {
  // read button states
  uint8_t btnState = 0;
  uint16_t fsr = calcPressure(analogRead(SensorPin));
  Serial.println(fsr);
  for (int i = 0; i < 8; i++) btnState |= (digitalRead(btnsPin[i]) << i);
  if (!btnState) noTone(buzzerPin);
  else {
    uint16_t pressure = calcPressure(fsr);
    pressure = pressureFilter.getVal(pressure);
    tone(buzzerPin, calcFreq(btnState, pressure));
    delay(10);
  }
}