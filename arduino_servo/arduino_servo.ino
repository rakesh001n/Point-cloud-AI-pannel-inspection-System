#include <Servo.h>

Servo rackServo;
const int servoPin = 9;

void setup() {
  Serial.begin(9600);
  rackServo.attach(servoPin);
  rackServo.write(0);
  Serial.println("Enter angle (0-180):");
}

void loop() {
  if (Serial.available() <= 0) return;
  int angle = Serial.parseInt();
  angle = constrain(angle, 0, 180);
  rackServo.write(angle);
  Serial.print("Servo moved to: ");
  Serial.println(angle);
  while (Serial.available() > 0) Serial.read();
}
