#include <ADNS3080.h>

#define PIN_RESET 9
#define PIN_CS    10

#define LED_MODE true
#define RESOLUTION false

ADNS3080<PIN_RESET, PIN_CS> sensor;

void setup() {
  // put your setup code here, to run once:
  sensor.setup(LED_MODE, RESOLUTION);
  pinMode(7, OUTPUT);
  Serial.begin(9600);
  digitalWrite(7, HIGH);
}

int x = 0, y = 0;

void loop() {
  // put your main code here, to run repeatedly:
  int8_t dx, dy;
  sensor.displacement(&dx, &dy);
  x += dx, y += dy;
  Serial.print("dx="), Serial.print(dx), Serial.print(" dy ="), Serial.print(dy);
  Serial.print(" x="), Serial.print(x), Serial.print(" y="), Serial.println(y);
}
