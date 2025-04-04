#define SensorPin A0

void setup() {
  Serial.begin(9600);
}

void loop() {
  uint16_t fsr = analogRead(SensorPin);
  Serial.println(fsr);
  delay(1);
}