#include <Arduino.h>

// Servo pin
const int servoPin = A1;

// Servo pulse widths in microseconds
const int SERVO_MIN = 0;   // 0 degrees
const int SERVO_MAX = 3000;   // 180 degrees

void setup() {
  pinMode(servoPin, OUTPUT);
  
  // Start at center position
  servoWrite(90);
  delay(1000);
}

// Send one servo pulse (call this repeatedly)
void servoWrite(int angle) {
  if (angle < 0)   angle = 0;
  if (angle > 180) angle = 180;

  int pulse = map(angle, 0, 180, SERVO_MIN, SERVO_MAX);

  digitalWrite(servoPin, HIGH);
  delayMicroseconds(pulse);
  digitalWrite(servoPin, LOW);
  delayMicroseconds(20000 - pulse);   // Complete ~20ms cycle (50Hz)
}

void loop() {
  // Sweep from 0° to 180° smoothly
  for (int angle = 0; angle <= 180; angle += 2) {
    servoWrite(angle);
    // Small extra delay makes movement smoother and less jerky
    delay(10);
  }

  // Sweep back from 180° to 0°
  for (int angle = 180; angle >= 0; angle -= 2) {
    servoWrite(angle);
    delay(10);
  }

  delay(400);   // Pause at minimum position
}