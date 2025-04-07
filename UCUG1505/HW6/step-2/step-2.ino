#define SensorPin A0
const int buzzerPin = 11;
const int forceConstant = 510;

void setup() {
  Serial.begin(9600);
}

void loop() {
  int sensorValue = analogRead(SensorPin);
  uint16_t force = (uint32_t)(1023 - sensorValue) * 510 / sensorValue;
  Serial.println(force);
  if (force <= 10) noTone(buzzerPin);
  else tone(buzzerPin, constrain(map(force, 10, 10000, 220, 1000), 220, 1000));
  delay(1); // Delay for 1 second
}