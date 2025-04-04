#define SensorPin A0
const int buzzerPin = 11;
const int resistence0 = 10000;

void setup() {
  Serial.begin(9600);
}

void loop() {
  int sensorValue = analogRead(SensorPin);
  int force = (1023 - sensorValue) * resistence0 / sensorValue;
  if (force <= 10) noTone(buzzerPin);
  else tone(buzzerPin, map(force, 10, 10000, 220, 10000));
  delay(1); // Delay for 1 second
}