#define SensorPin A0
#define AutoMode
const int buzzerPin = 11;
const int musicPin = 12;
const uint32_t forceConstant = 510;
const uint8_t btnsPin[8] = {2, 3, 4, 5, 7, 8, 9, 10};

// add a filter
#define MidAvgFilterSz  (8u)
template<typename T>
struct MidAvgFilter {
  T buf[MidAvgFilterSz];
  uint8_t idx = 0;
  MidAvgFilter() {
    this->idx = 0;
    memset(this->buf, 0, sizeof(this->buf));
  }
  T getVal(T newData) {
    uint8_t i, j;
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
      return tmpArr[(MidAvgFilterSz | 1) >> 1];
    else return (tmpArr[MidAvgFilterSz >> 1] + tmpArr[(MidAvgFilterSz >> 1) | 1]) >> 1;
  }
};

MidAvgFilter<uint32_t> pressureFilter;

const uint32_t freqMap[16] = {261, 294, 330, 349, 392, 440, 494, 523, 587, 659, 698, 783, 880, 988, 1046};

const uint8_t music[] = {
  2, 
  4, 4, 4, 2, 2, 2, 
  3, 2, 3, 4, 5, 3, 
  5, 5, 5, 7, 6, 4, 
  9, 9, 9, 11, 11, 11, 
  7, 7, 7, 7, 9, 9, 9, 9, 4, 
  5, 5, 7, 10, 9, 7, 7, 7, 6, 
  5, 5, 8, 7, 6, 4, 
  9, 9, 9, 11, 11, 11
};

#ifdef AutoMode
const uint8_t gap[] = {
  2, 
  4, 4, 8, 4, 4, 8, 
  4, 4, 4, 4, 8, 8,
  4, 4, 4, 4, 8, 8,
  4, 4, 8, 4, 4, 8, 
  4, 4, 4, 4, 4, 4, 4, 3, 1,
  4, 4, 4, 4, 4, 4, 4, 2, 2,
  4, 4, 4, 4, 8, 8,
  4, 4, 8, 4, 4, 8,
};
uint16_t curGap = 0;
#else
uint8_t lstMusicBtnState = 0;
#endif

uint16_t musicIdx = sizeof(music) - 1;

inline uint32_t calcPressure(uint32_t fsr) {
  return (uint32_t)(1023 - fsr) * forceConstant / fsr;
}

uint16_t calcFreq(uint8_t btns, uint32_t pressure) {
  uint16_t res = 0;
  for (int i = 0; i < 8; i++) if (btns & (1u << i)) 
    res += freqMap[i];
  res += map(constrain(pressure, 0, 70000), 0, 70000, 0, 100);
  return res;
}

void setup() {
  Serial.begin(9600);
  for (int i = 0; i < 8; i++) pinMode(btnsPin[i], INPUT);
  pinMode(musicPin, INPUT);
  pinMode(buzzerPin, OUTPUT);
  #ifndef AutoMode
  musicIdx = sizeof(music) - 1;
  lstMusicBtnState = 0;
  #else
  musicIdx = 0;
  curGap = 0;
  #endif
}

void loop() {
  // read button states
  uint8_t btnState = digitalRead(musicPin);
  #ifndef AutoMode
  if (musicBtnState != lstMusicBtnState) {
    if (musicBtnState) {
      musicIdx = (musicIdx + 1) % sizeof(music);
    }
    lstMusicBtnState = musicBtnState;
  }
  if (musicBtnState) {
    // find the tone
    tone(buzzerPin, freqMap[music[musicIdx]]);
  #else
  if (btnState) {
    curGap++;
    if (gap[musicIdx] < curGap) {
      musicIdx = (musicIdx + 1) % sizeof(music);
      curGap = 0;
      tone(buzzerPin, freqMap[music[musicIdx]]);
      delay(gap[musicIdx] * 70);
    } else {
      noTone(buzzerPin);
      delay(10);
    }
  #endif
  } else {
    uint32_t fsr = calcPressure(analogRead(SensorPin));
      Serial.println(fsr);
    for (int i = 0; i < 8; i++) btnState |= (digitalRead(btnsPin[i]) << i);
    if (!btnState) { noTone(buzzerPin); delay(10); }
    else {
      fsr = pressureFilter.getVal(fsr);
      tone(buzzerPin, calcFreq(btnState, fsr));
      delay(10);
    }
    
  }
}