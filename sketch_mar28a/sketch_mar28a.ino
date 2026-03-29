#include <Arduino.h>

// Servo on analog-capable pin (bit-banged PWM)
const int servoPin = A1;

// Servo pulse widths in microseconds (matches your original map range)
const int SERVO_MIN = 0;
const int SERVO_MAX = 3000;

int targetAngle = 90;

void setup() {
  pinMode(servoPin, OUTPUT);
  Serial.begin(115200);
  delay(300);
  // Hold center briefly
  for (int i = 0; i < 50; i++) {
    servoWrite(targetAngle);
  }
  delay(300);
}

// One ~20ms servo cycle (50Hz). Call often to hold position.
void servoWrite(int angle) {
  if (angle < 0) angle = 0;
  if (angle > 180) angle = 180;

  int pulse = map(angle, 0, 180, SERVO_MIN, SERVO_MAX);

  digitalWrite(servoPin, HIGH);
  delayMicroseconds(pulse);
  digitalWrite(servoPin, LOW);
  delayMicroseconds(20000 - pulse);
}

void loop() {
  if (Serial.available() > 0) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length() > 0) {
      int v = line.toInt();
      targetAngle = constrain(v, 0, 180);
    }
  }
  servoWrite(targetAngle);
}
